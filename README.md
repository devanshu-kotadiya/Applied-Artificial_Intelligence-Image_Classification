# Applied-Artificial_Intelligence-Image_Classification

### Main Objective
The main goal of this project is to study a Machine + Deep Learning (ML+DL) task using Decision Trees, Random Forests, Boosting, and Convolutional Neural Networks (CNN) to classify the museum images in two categories (indoor, outdoor), applying supervised and semi-supervised learning Classification.

### Requirements to Run the Code
<pre><code>pip install torch torchvision scikit-learn matplotlib seaborn pandas
</code></pre>
<br>
You’ll also need access to a GPU if available for faster training (optional but recommended).<br>

### How to Obtain the Dataset<br>
The dataset is assumed to be pre-uploaded in the /kaggle/input/museum/Training path (Kaggle Notebook environment).<br>
<br>
If you’re not using Kaggle, you can manually download and structure the dataset from a source like:<br>
	•	Custom indoor/outdoor dataset<br>
	•	Places365 (subset): http://places2.csail.mit.edu/download.html<br>

 ### Source Code
 Your package includes:<br>
	•	CNNClassifier: CNN architecture class<br>
	•	train_and_evaluate(): Training loop with performance evaluation<br>
	•	Experiment runner that varies hyperparameters<br>
  Metrics computation using:<br>
	•	accuracy_score<br>
	•	precision_score<br>
	•	recall_score<br>
	•	f1_score<br>
	•	confusion_matrix<br>
	Visualization tools for confusion matrix and tracking performance
