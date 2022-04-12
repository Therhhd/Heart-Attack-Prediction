{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d782e2c0-2954-45a4-b97b-7b3391355a2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "This file includes all of the methods required to create, train and test the LSTM model\n",
    "        that is used to predict episodes of VT.\n",
    "        Adapted from: https://github.com/aurotripathy/lstm-ecg-wave-anomaly-detect/blob/master/lstm-ecg-wave-anomaly-detect.py\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7d03899-223f-45b7-b9c4-e7bbc331e7a5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from readData import CARDIAC_EVENTS, TYPES_OF_EVENTS_DICT\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f78bc210-973c-4038-9b95-f6b732cc186a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from keras.layers.core import Dense, Activation, Dropout\n",
    "from keras.layers.recurrent import LSTM\n",
    "from keras.models import Sequential\n",
    "from keras.optimizers import SGD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1386fdd8-4554-4fba-a2e0-b01a5e8d607d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Hyperparameters\n",
    "# batch_size = 32:\n",
    "epoch = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6989fdc-6a7a-4943-bfe6-b6745e5ed790",
   "metadata": {},
   "outputs": [],
   "source": [
    "ef createModel(numOfBeats, numOfReadings, numOfOutcomes):\n",
    "        '''\n",
    "                Creates, compiles and returns an LSTM model that can be used to predict episodes of VT.\n",
    "        '''\n",
    "        model = Sequential()\n",
    "\n",
    "        model.add(LSTM(32, input_shape=(numOfBeats, numOfReadings), return_sequences=True))\n",
    "        model.add(Dropout(0.2))\n",
    "\n",
    "        model.add(LSTM(32, return_sequences=True))\n",
    "        model.add(Dropout(0.2))\n",
    "\n",
    "        model.add(LSTM(32, return_sequences=False))\n",
    "        model.add(Dropout(0.2))\n",
    "\n",
    "        model.add(Dense(numOfOutcomes))\n",
    "        model.add(Activation('softmax'))\n",
    "\n",
    "        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)\n",
    "        model.compile(loss='categorical_crossentropy', optimizer='adam')\n",
    "        return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88be4225-7f21-4912-8d5c-0230f8f08453",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluateModel(rawPredictedY, rawActualY, verboseFile):\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f384892-2f15-494a-88eb-cbec5f34120d",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "        Evaluates a LSTM model that can be used to predict episodes of VT.\n",
    "        \n",
    "        Returns a tuple of the form ([sensitivity], [specificity]) where [sensitivity] is the sensitivity\n",
    "        of the model's ability to predict episodes of VT and [specificity] is the specificity of the model's\n",
    "        ability to predict episodes of VT.\n",
    "    \n",
    "    '''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56debede-7ec7-4500-8958-c19e10dfd063",
   "metadata": {},
   "outputs": [],
   "source": [
    "numOfTesting, numOfOutcomes) = np.shape(rawActualY)\n",
    "    \n",
    "    # Convert probabilities in rawPredictedY into classifications.\n",
    "    predictedY = np.zeros((numOfTesting, 1), dtype=np.int16)\n",
    "    for testingIndex in range(numOfTesting):\n",
    "        \n",
    "        maxVal = 0\n",
    "        maxOutcomeIndex = 0\n",
    "        for outcomeIndex in range(numOfOutcomes):\n",
    "            val = rawPredictedY[testingIndex, outcomeIndex]\n",
    "            if (val > maxVal):\n",
    "                maxVal = val\n",
    "                maxOutcomeIndex = outcomeIndex\n",
    "        \n",
    "        predictedY[testingIndex, 0] = maxOutcome"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e133813-78ef-4061-8f89-f704771fea5a",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Convert binary things in rawPredictedY into classficiations.\n",
    "    actualY = np.zeros((numOfTesting, 1), dtype=np.int16)\n",
    "    for testingIndex in range(numOfTesting):\n",
    "        for outcomeIndex in range(numOfOutcomes):\n",
    "            if (rawActualY[testingIndex, outcomeIndex] == 1):\n",
    "                actualY[testingIndex, 0] = outcomeIndex\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a112614-4cfa-412b-91d2-14ac74ea98c5",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Set up dictionaries that keep track of the actual and predicted numbers\n",
    "    # of each cardiac event.\n",
    "    numOfEachAuxPredicted = {}\n",
    "    numOfEachAuxActual = {}\n",
    "    for aux in TYPES_OF_EVENTS_DICT:\n",
    "        numOfEachAuxPredicted[aux] = 0\n",
    "        numOfEachAuxActual[aux] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9891bd5f-1fbe-4726-80ab-f7f1b92f16ee",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Compute entries in dictionaries that keep track of the actual and predicted\n",
    "    # numbers of each cardiac event.\n",
    "    for index in range(numOfTesting):\n",
    "        predictedAux = CARDIAC_EVENTS[predictedY[index, 0]]\n",
    "        actualAux = CARDIAC_EVENTS[actualY[index, 0]]\n",
    "            \n",
    "        numOfEachAuxPredicted[predictedAux] = numOfEachAuxPredicted[predictedAux] + 1\n",
    "        numOfEachAuxActual[actualAux] = numOfEachAuxActual[actualAux] + 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f720fc0-a67a-40f4-91a7-4bb66c1d3dee",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Print entries in dictionaries that keep track of the actual and predicted\n",
    "    # numbers of each cardiac event.\n",
    "    for aux in TYPES_OF_EVENTS_DICT:\n",
    "        verboseFile.write('Cardiac Event: ' + str(TYPES_OF_EVENTS_DICT[aux]) + '\\n')\n",
    "        verboseFile.write('\\tActual: ' + str(numOfEachAuxActual[aux]) + '\\n')\n",
    "        verboseFile.write('\\tPredicted: ' + str(numOfEachAuxPredicted[aux]) + '\\n')\n",
    "        verboseFile.write('')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b56c161d-08b4-40e0-b77e-6909ee73358f",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Calculate specificity and sensitivity.\n",
    "    tn = 0\n",
    "    n = 0\n",
    "    tp = 0\n",
    "    p = 0\n",
    "    numOfCorrect = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d827502-e4a5-4ed6-b389-84715cc3d709",
   "metadata": {},
   "outputs": [],
   "source": [
    "numForVT = CARDIAC_EVENTS.index('(VT')\n",
    "    for i in range(numOfTesting):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0874990-d071-4cd9-b7d7-01f964c4afe1",
   "metadata": {},
   "outputs": [],
   "source": [
    "if (actualY[i, 0] == predictedY[i, 0]):\n",
    "            numOfCorrect = numOfCorrect + 1\n",
    "            \n",
    "        # Positives\n",
    "        if (actualY[i, 0] == numForVT):\n",
    "            p = p + 1\n",
    "                    \n",
    "            # True positives\n",
    "            if (predictedY[i, 0] == numForVT):\n",
    "                tp = tp + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45daa5ac-6eb7-4429-bf07-d25e7b4cc529",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "        # Negatives\n",
    "        else:\n",
    "            n = n + 1\n",
    "                    \n",
    "            # True negatives\n",
    "            if (predictedY[i, 0] != numForVT):\n",
    "                tn = tn + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "558e4259-c05f-4485-83ee-61c1da543b5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "  verboseFile.write('Num of positives: ' + str(p) + '\\n')\n",
    "    verboseFile.write('Num of negatives: ' + str(n) + '\\n')\n",
    "    verboseFile.write('Num of true positives: ' + str(tp) + '\\n')\n",
    "    verboseFile.write('Num of true negatives: ' + str(tn) + '\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "429e2c4a-2692-44ed-9dcf-ae82c35a7948",
   "metadata": {},
   "outputs": [],
   "source": [
    "    verboseFile.write('Num of correct: ' + str(numOfCorrect) + '\\n')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "042bf689-49fc-416b-a8be-be97a8d321bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "if (p == 0):\n",
    "        sensitivity = -1\n",
    " else:\n",
    "        sensitivity = float(tp) / float(p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ccd6f9c-96ca-4378-85c0-5ebc170458a8",
   "metadata": {},
   "outputs": [],
   "source": [
    " if (n == 0):\n",
    "        specificity = -1\n",
    "    else:\n",
    "        specificity = float(tn) / float(n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9d5dd10-e3d5-4e7a-ab13-cb589c0e7221",
   "metadata": {},
   "outputs": [],
   "source": [
    "    verboseFile.write(\"Sensitivity: \" + str(sensitivity) + '\\n')\n",
    "    verboseFile.write(\"Specificity: \" + str(specificity) + '\\n')\n",
    "    verboseFile.write('\\n\\n')    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53f3ad71-55b8-4dc0-8e74-2d2594c68f35",
   "metadata": {},
   "outputs": [],
   "source": [
    "return (sensitivity, specificity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ccdc300-b1f4-4033-b7a1-99c095740fb9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def runModel(trainingX, trainingY, testingX, testingY, verboseFile):\n",
    "    '''\n",
    "        Creates, trains and evaluates an LSTM model that can be used to predict episodes of VT.\n",
    "        \n",
    "        Returns a tuple of the form ([sensitivity], [specificity]) where [sensitivity] is the sensitivity\n",
    "        of the model's ability to predict episodes of VT and [specificity] is the specificity of the model's\n",
    "        ability to predict episodes of VT.\n",
    "    \n",
    "    '''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09e2c9a2-c09c-436a-a833-91e6ba4ba811",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Create the LSTM model.\n",
    "    print (\"Creating...\")\n",
    "    numOfOutcomes = len(CARDIAC_EVENTS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "213231ba-479b-47a9-8304-9ce6986948d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "for cardiacEvent in CARDIAC_EVENTS:\n",
    "         print (cardiacEvent)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18ab013d-cf27-42b8-9089-573f5b738743",
   "metadata": {},
   "outputs": [],
   "source": [
    "for cardiacEvent in CARDIAC_EVENTS:\n",
    "         print (cardiacEvent)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9815d37f-66b4-43eb-b831-40636ecead43",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the LSTM model.\n",
    "    print (\"Training...\")\n",
    "    model.fit(x = trainingX, y = trainingY, nb_epoch = epoch, validation_split = 0.05)\n",
    "    print (\"Completed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3d6dfbf-e0c7-4221-a6fd-3263c83fce99",
   "metadata": {},
   "outputs": [],
   "source": [
    "   # Evaluate the LSTM model.\n",
    "    print (\"Evaluating...\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f0f17cc-b80d-4f93-97d8-7a0760ef97b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "    #Evaluate the LSTM model.\n",
    "    print (\"Evaluating...\")\n",
    "    rawPredictedY = model.predict(testingX)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3658daf-3401-415a-a89f-fe539701801c",
   "metadata": {},
   "outputs": [],
   "source": [
    " for outcome in range(numOfOutcomes):\n",
    "          print rawPredictedY[0, outcome]\n",
    "    (sensitivity, specificity) = evaluateModel(rawPredictedY, testingY, verboseFile)\n",
    "    print (\"Completed!\")\n",
    "\n",
    "    return (sensitivity, specificity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "131202fd-6667-4303-864a-d5521a4e8120",
   "metadata": {},
   "outputs": [],
   "source": [
    " return (sensitivity, specificity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04aa726d-b6d2-438c-87cd-18ec63f764d1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc802905-1474-4300-852d-fe60a299cc17",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "637e0d21-e1f2-425a-8836-99c533fdc340",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6335f64-e312-41a2-92d5-36a9e5846754",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6efd8cec-205e-424a-936c-684cf3f38f97",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0f57e21-665e-4656-8489-4eed19c23807",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c410384-4b28-4a12-874f-aa48ca2118a7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "523acc6a-39e9-4674-8c23-98fc9c54c9e1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e966d82-b911-45b5-a5e6-e4eea3a171ba",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccb007b4-9a59-430b-aed7-8bc480f40d9a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae122f60-4aee-4906-8ed1-20c6f7f4b2f2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8de11ffe-f6f4-46e0-b15f-34711ff70ed8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c40079a-ce93-4089-8b22-36707e9a8630",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a10287e-c87b-4e54-b975-5edff5ab0f52",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03014076-164e-488d-b796-89777a8b1419",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7fc9e2b4-5327-4cb9-b693-4eeb5821e745",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b433d313-ee85-4eea-a8d0-3371628f0aad",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "490e0aab-44ed-4d60-ad62-d1d6f4b013b8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebdf4783-5e2e-472a-9c2f-2f66be1478ee",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8974a4c-4b80-4e5b-ad57-d28dc9ceb861",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21389ae4-bacf-4d75-ae7d-8198b9315273",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d09a996d-3add-4d59-9d13-63b721eff094",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38978d8e-3f32-4667-b580-8f6d6a0a3e97",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db4f80ee-e74b-44ab-97f8-68d9470ee2e1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "186e5431-54ac-4e35-8a66-c4b5b968e814",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bb5f94c-ad14-48e1-a7d5-ad59ebd31b74",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca0f6bcf-6e86-4930-90f1-ae79f51f025d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "507e8432-21ec-496f-bdeb-0e3e0e78e0ee",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e838b36b-ec50-4573-94e7-91f97ca2e602",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "faa9ee2b-f13f-4bb6-916a-fc4ef6d10d88",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9f8b249-e16c-4bb7-8a7f-7a238393fa19",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd6a0589-946c-41a0-a676-c5fb95c834ca",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09f3993e-3bae-4fea-86cc-6cec86806af4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca66be97-5714-4d56-803e-b50fabf230a0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3281a38b-98ad-4389-bbb4-c877ea394a29",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9e48f40-baad-476e-b594-cfd94579b46b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa7c7115-0e70-4fc5-a285-6c781adbaa4f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c11bb5b7-07ef-499e-bcb5-f43d6dce50c4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d83c1fc-8c3d-4b4e-8d69-b9bb34bddf6c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18b8ede1-aefc-4217-9f48-9056b52edfe7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9936c607-b7a3-41f0-847b-afb15b323bfa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a112a5ea-b962-4a5e-8abe-ab730027395e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d8e6ff2-fd55-49c7-8933-924d24db20eb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47529908-425d-45c6-97ed-6c4192251bbf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64893720-bf70-4572-bb0c-2c61d36986b5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ce80a78-ce41-4e96-89c5-c7859498a603",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2c1be8b-b435-44fd-9534-6f99b4468345",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
