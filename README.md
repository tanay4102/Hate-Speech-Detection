# Hate-Speech-Detection
Task 1 
1.train-preprocessing and test-preprocessing files are the preprocessed files used in the code.
2.The code uses fasttext module,numpy,scikit and spacy. Everything is followed according to the instructions mention in the PDF.
3.SVM portion takes time. The whole code should be done in 10-15 minutes.
Task 2
1.Things needed : Python SciPy ideally with Python 3. Keras (2.2 or higher) installed with either the TensorFlow or Theano backend.The tutorial also assumes you have scikit-learn and NumPy.
2.Methodology : 
a)Data Preperations : The tweets were first split into hateful and non-hateful.All of these files are stored in twt_sentoken folder. The sub folders are hate,nohate and test. Hate and No Hate contain files from training set seperated, test are the 5000 predictions. The training data is further segregated into a train set and a test set (90-10 split for both hate and nohate). The files that are being used for training start with train and the other files start with test. 
b)Preprocessing : Split tokens on white space. Remove all punctuation from words. Remove all words that are not purely comprised of alphabetical characters. Remove all words that are known stop words (nltk).
Remove all words that have a length <= 1 character. This is done in the function clean_doc.
c)Defining Vocabulary : Developed a vocabulary as a Counter, which is a dictionary mapping of words and their counts that allow us to easily update and query.add_doc_to_vocab and process_doc.
d)Train Embedding Layer : Tokeniser class from Keras, after fittin on texts from train_docs. Ensured all docs of same length by padding smaller words to the max length. 
e)Defining Model : Using Keras Sequential model and training it on the F1 score.
Ouput : Details of the layer, Epochs simulations and F1 score on test set.

