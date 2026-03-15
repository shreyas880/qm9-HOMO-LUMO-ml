<h1>ML Prediction and Interpretation of HOMO-LUMO Gaps w/ QM9 Dataset</h1>

<h2>Overview:</h2>
<p>
This is a project that predicts molecular electronic properties utilizing machine learning.<br>
Using the QM9 dataset (https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9/data), molecular structures were encoded with Morgan fingerprints and the HOMO-LUMO gap was predicted using a Random Forest model.<br>
Beyond raw numbers, the model's predictions were interpreted by identifying important fingerprint features by relating them to their bit representations. This was then further mapped to their chemical sub-structures allowing further introspection of motifs that could possibly influence electronic properties.
</p>

<h2>Motivation:</h2>
<p>
I was motivated to apply my new-found learnings into the field of my liking. The HOMO-LUMO gap was something I have learnt about, on multiple occasions and to my understanding is quite an essential property in predicting reactivity of compounds and therefore was quite enticing to work on.<br>
Just a project to understand the applications of Machine Learning in Chemistry as it is a very intriguing topic that I would love to indulge myself into, as a future career path.
</p>

<h2>Dataset:</h2>
<p>The project uses the QM9 dataset, which contains approx. 134k data samples*. The samples are small organic molecules with some important properties computed using Density Function Theory (DFT).<br>
Each molecule includes properties such as:
  <ul>
    <li>HOMO Energy</li>
    <li>LUMO Energy</li>
    <li>Dipole Moment</li>
    <li>Polarizability</li>
    <li>SMILES representation of the compound</li>
  </ul>
<br>
The original dataset is then parsed and converted into a csv keeping only relevant data and discarding the rest.<br> 
50000 samples were used to train the model in this project, of which, about 10k samples were used in testing/validation of models.<br>
*The dataset is not included in the repository due to size.
</p>

<h2>Machine Learning Model:</h2>
<p>
The final project model is a RandomForest regression model.<br>
The following models' performances were tested on 5k samples:
<ul>
  <li>Random Forest</li>
  <li>Linear Regression</li>
  <li>Gradient Boost</li>
</ul>
The Random Forest model seemed to be the best for this task with a reasonably low Mean Absolute Error (MAE), and a high enough R2 score, which led me to proceed with this specific model.  
<br>
<h3>Final Model Parameters:</h3>
<ul>
  <li>n_estimators = 100</li>
  <li>min_samples_split = 5</li>
  <li>criterion = squared_error</li>
  <li>n_jobs = -1 (for parallel & faster processing)</li>
</ul>
</p>

<h2>Results</h2>
<h4>Mean Absolute Error (MAE) ≈ 0.008 eV<br>
R² ≈ 0.93</h4>
<p>
Prediction vs true values:<br>
The results show strong predictive performance for this relatively simple model.
</p>

<h2>Interpretation of Predicted Results:</h2>
<p>
  Feature Importance Analysis was used to find the fingerprint bits that were most strongly associated with changes in the HOMO-LUMO Gap.<br>
  For each fingerprint bit:
  <ol>
    <li>Molecules activating the bit were identified</li>
    <li>Relevant parts of the structure were highlighted</li>
    <li>Multiple example molecules were visualised</li>
  </ol>
  This allows my model to be interpreted in terms of structure-property relationships.<br>
  Also, for the important bits, the distribution of HOMO-LUMO gaps was compared b/w molecules.
</p>

<h2>Tools Used:
<ul>
  <li>RDKit</li>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Pyplot</li>
  <li>Scikit learn</li>
</ul>
</h2>
