import io

from PyPDF2 import PdfReader
from flask import Flask, request
from tensorflow import keras
import numpy as np
import json
import PyPDF2

app = Flask(__name__)

minMCHC = 18.07
maxMCHC = 168.0
minHCT = 7.21
maxHCT = 67.4
minHGB = 2.99
maxHGB = 22.45
minMCV = 49.93
maxMCV = 125.3
minPLT = 13
maxPLT = 1770
minFERRITTE = 0.5
maxFERRITTE = 27332.0
minB12 = 30.0
maxB12 = 33880.0
minFOLATE = 0.55
maxFOLATE = 50.25

minTB = 0.4
maxTB = 75.0
minSGPT = 10
maxSGPT = 2000
minSGOT = 10
maxSGOT = 4929
minALB = 0.9
maxALB = 5.5

minWBC = 0.57
maxWBC = 246.7
minRBC = 0.87
maxRBC = 7.5
minRDW = 9.72
maxRDW = 39.4

liver_model = keras.models.load_model('models/liver.h5')
CBCA_model = keras.models.load_model('models/CBC_Advance.h5')
CBC_model = keras.models.load_model('models/cbc.h5')


@app.route('/liver_pred', methods=['POST'])
def liver_pred():
    input_data = request.get_json()

    TB = (input_data['TotalBilirubin'] - minTB) / (maxTB - minTB)
    Sgpt = (input_data['SgptAlamineAminotransferase'] - minSGPT) / (maxSGPT - minSGPT)
    Sgot = (input_data['SgotAspartateAminotransferase'] - minSGOT) / (maxSGOT - minSGOT)
    ALB = (input_data['ALBAlbumin'] - minALB) / (maxALB - minALB)
    gender = input_data['gender_dummy']

    input_list = [TB, Sgpt, Sgot, ALB, gender]

    prediction = liver_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)

    predicted = None
    if class_labels[0] == 0:
        predicted = "Nothing"
    elif class_labels[0] == 1:
        predicted = "Virus C"
    elif class_labels[0] == 2:
        predicted = "Gallbladder"
    elif class_labels[0] == 3:
        predicted = "Virus A"
    elif class_labels[0] == 4:
        predicted = "Fatty Liver"

    data = {
        "status": True,
        "message": "Disease is predicted",
        "data": {
            "disease": predicted
        }
    }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/CBCA_pred', methods=['POST'])
def CBCA_pred():
    input_data = request.get_json()

    MCHC = (input_data['MCHC'] - minMCHC) / (maxMCHC - minMCHC)
    HCT = (input_data['HCT'] - minHCT) / (maxHCT - minHCT)
    HGB = (input_data['HGB'] - minHGB) / (maxHGB - minHGB)
    MCV = (input_data['MCV'] - minMCV) / (maxMCV - minMCV)
    PLT = (input_data['PLT'] - minPLT) / (maxPLT - minPLT)
    FERRITTE = (input_data['FERRITTE'] - minFERRITTE) / (maxFERRITTE - minFERRITTE)
    B12 = (input_data['B12'] - minB12) / (maxB12 - minB12)
    FOLATE = (input_data['FOLATE'] - minFOLATE) / (maxFOLATE - minFOLATE)
    GENDER = input_data['GENDER']

    # X=np.asarray(df[['MCHC','HCT','HGB','MCV','PLT','FERRITTE','B12','FOLATE','GENDER']].values.tolist())

    input_list = [MCHC, HCT, HGB, MCV, PLT, FERRITTE, B12, FOLATE, GENDER]

    prediction = CBCA_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)

    predicted = None
    if class_labels[0] == 0:
        predicted = "Nothing"
    elif class_labels[0] == 1:
        predicted = "HGB anemia"
    elif class_labels[0] == 2:
        predicted = "iron anemia"
    elif class_labels[0] == 3:
        predicted = "Folate anemia"
    elif class_labels[0] == 4:
        predicted = "B12 anemia"

    data = {
        "status": True,
        "message": "Disease is predicted",
        "data": {
            "disease": predicted
        }
    }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/CBC_pred', methods=['POST'])
def CBC_pred():
    input_data = request.get_json()

    WBC = (input_data['WBC'] - minWBC) / (maxWBC - minWBC)
    RBC = (input_data['RBC'] - minRBC) / (maxRBC - minRBC)
    HGB = (input_data['HGB'] - minHGB) / (maxHGB - minHGB)
    MCV = (input_data['MCV'] - minMCV) / (maxMCV - minMCV)
    RDW = (input_data['RDW'] - minRDW) / (maxRDW - minRDW)
    PLT = (input_data['PLT'] - minPLT) / (maxPLT - minPLT)
    GENDER = input_data['GENDER']

    # X=np.asarray(df[['MCHC','HCT','HGB','MCV','PLT','FERRITTE','B12','FOLATE','GENDER']].values.tolist())

    input_list = [WBC, RBC, HGB, MCV, RDW, PLT, GENDER]
    prediction = CBC_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)
    predicted = None
    if class_labels[0] == 0:
        predicted = "Nothing"
    elif class_labels[0] == 1:
        predicted = "hemolytic"
    elif class_labels[0] == 2:
        predicted = "macrocytic"
    elif class_labels[0] == 3:
        predicted = "penectopenia"
    elif class_labels[0] == 4:
        predicted = "microcytic hypochromic"

    data = {
        "status": True,
        "message": "Disease is predicted",
        "data": {
            "disease": predicted
        }
    }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/CBC_read', methods=['POST'])
def CBC_read():
    file = request.files['file']
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    to_be_find = ['Haemoglobin', 'Red Cells Count', 'Haematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 'Platelets Count',
                  'Total Leucocytic Count'
        , 'Basophils %', 'Eosinophils %', 'Neutrophils %', 'Lymphocytes %', 'Monocytes %', 'Neutrophils absolute count',
                  'Lymphocytes absolute count', 'Monocytes absolute count', 'Eosinophils absolute count',
                  'Basophils absolute count']
    parameters = []
    for parameter in to_be_find:
        parameter_value = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Find the index of the parameter in the text
            index = text.find(parameter)
            if index != -1:
                line = text[index:text.find('\n', index)]
                words = line.split()
                # Find the index of the parameter value in the line
                value_index = words.index(parameter.split()[-1]) + 1
                parameter_value = text[index:].split()[value_index]
                parameters.append(parameter_value)
                break  # Exit the page loop if the parameter is found
            else:
                parameters.append(-1)

    param_dict = {}
    for name, value in zip(to_be_find, parameters):
        param_dict[name] = value

    if all(value == -1 for value in param_dict.values()):
        data = {
            "status": False,
            "message": "The file content is not proper, try to fill manually",
            "data": param_dict
        }
    else:
        data = {
            "status": True,
            "message": "CBC is added",
            "data": param_dict
        }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/liver_read', methods=['POST'])
def liver_read():
    file = request.files['file']
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    to_be_find = ['GammaGT', 'Bilirubin_Total', 'Bilirubin_Direct', 'AST', 'ALT', 'Alk', 'TotalProtein', 'Albumin']

    parameters = []
    for parameter in to_be_find:
        parameter_value = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Find the index of the parameter in the text
            index = text.find(parameter)
            if index != -1:
                line = text[index:text.find('\n', index)]
                words = line.split()
                # Find the index of the parameter value in the line
                value_index = words.index(parameter.split()[-1]) + 1
                parameter_value = text[index:].split()[value_index]
                parameters.append(parameter_value)
                break  # Exit the page loop if the parameter is found
            else:
                parameters.append(-1)

    param_dict = {}
    for name, value in zip(to_be_find, parameters):
        param_dict[name] = value
    if all(value == -1 for value in param_dict.values()):
        data = {
            "status": False,
            "message": "The file content is not proper, try to fill manually",
            "data": param_dict
        }
    else:
        data = {
            "status": True,
            "message": "Liver is added",
            "data": param_dict
        }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/Renal_read', methods=['POST'])
def Renal_read():
    file = request.files['file']
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    to_be_find = ['Urea', 'CreatinineInSerum', 'UricAcid']
    parameters = []
    for parameter in to_be_find:
        parameter_value = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Find the index of the parameter in the text
            index = text.find(parameter)
            if index != -1:
                line = text[index:text.find('\n', index)]
                words = line.split()
                # Find the index of the parameter value in the line
                value_index = words.index(parameter.split()[-1]) + 1
                parameter_value = text[index:].split()[value_index]
                parameters.append(parameter_value)
                break  # Exit the page loop if the parameter is found
            else:
                parameters.append(-1)

    param_dict = {}
    for name, value in zip(to_be_find, parameters):
        param_dict[name] = value
    if all(value == -1 for value in param_dict.values()):
        data = {
            "status": False,
            "message": "The file content is not proper, try to fill manually",
            "data": param_dict
        }
    else:
        data = {
            "status": True,
            "message": "Renal is added",
            "data": param_dict

        }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/stool_read', methods=['POST'])
def Stool_read():
    file = request.files['file']
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    to_be_find = ['Color', 'Consistency', 'FoodParticles', 'Mucus', 'Blood', 'Starch', 'Muscle fibers', 'Vegetables',
                  'Protozoa', 'Ciliates']
    parameters = []
    for parameter in to_be_find:
        parameter_value = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Find the index of the parameter in the text
            index = text.find(parameter)
            if index != -1:
                line = text[index:text.find('\n', index)]
                words = line.split()
                # Find the index of the parameter value in the line
                value_index = words.index(parameter.split()[-1]) + 1
                parameter_value = text[index:].split()[value_index]
                parameters.append(parameter_value)
                break  # Exit the page loop if the parameter is found
            else:
                parameters.append(-1)

    param_dict = {}
    for name, value in zip(to_be_find, parameters):
        param_dict[name] = value
    if all(value == -1 for value in param_dict.values()):
        data = {
            "status": False,
            "message": "The file content is not proper, try to fill manually",
            "data": param_dict
        }
    else:
        data = {
            "status": True,
            "message": "Stool is added",
            "data": param_dict

        }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


@app.route('/urine_read', methods=['POST'])
def Urine_read():
    file = request.files['file']
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    to_be_find = ['Color', 'Clarity', 'Specific Gravity', 'UrinePH', 'Protein', 'Glucose', 'Ketone', 'Urine bilirubin',
                  'Nitrite', 'Crystals', 'Casts']
    parameters = []
    for parameter in to_be_find:
        parameter_value = None
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Find the index of the parameter in the text
            index = text.find(parameter)
            if index != -1:
                line = text[index:text.find('\n', index)]
                words = line.split()
                # Find the index of the parameter value in the line
                value_index = words.index(parameter.split()[-1]) + 1
                parameter_value = text[index:].split()[value_index]
                parameters.append(parameter_value)
                break  # Exit the page loop if the parameter is found
            else:
                parameters.append(-1)

    param_dict = {}
    for name, value in zip(to_be_find, parameters):
        param_dict[name] = value
    if all(value == -1 for value in param_dict.values()):
        data = {
            "status": False,
            "message": "The file content is not proper, try to fill manually",
            "data": param_dict
        }
    else:
        data = {
            "status": True,
            "message": "Urine is added",
            "data": param_dict

        }

    # Convert to JSON string
    json_data = json.dumps(data)
    return json_data


if __name__ == '__main__':
    app.run()
