# PriberamChallenge

**Model**
The implemented solution for the document identification problem uses an unsupervised Bayesian approach. The words in each document are assumed to be drawn from the same Categorical word distribution. In other others, the model assumes that the paragraphs of a document use a small set of the vocabulary, and, thus, by finding changes in the vocabulary we can recover the documents. The Categorical word distributions are drawn from a Dirichlet prior. Finally, there is also a document length prior, which allows expressing our beliefs on how long documents are. The full generative process of the model uses the document length prior to decide if a new documents is going to start. If we are not starting a new document, the words for the next paragraph are drawn from the same word distribution as the previous paragraph. When starting a new document, a new word distribution is drawn from the Dirichlet prior.

**Inferencce**
The inference procedure uses a MAP approach. This is done by deriving from the previous model specification a mathematical expression for the likelihood of document start assignments to paragraphs. The word distribution variables can be integrated out, leaving only the document start labels as hidden variables. In this context, we only need to find the document start assignment configuration that yields the highest likelihood. Depending on the size of the dataset, a full exploration of the document start assignment space can be done. The size of the provided development set is already too big for the previous approach to be viable. Therefore, an approximate algorithm was implemented instead. The algorithm goes through each paragraph and computes the likelihood of assigning the paragraph to the previous document or starting a new one. During each iteration, the algorithm keeps track of the top-b assignment configurations that have the highest likelihood (a beam search feature). This allows recovering from early mistakes. To further improve the efficiency of the inference algorithm, the input data is processed on a chunk basis.

**Evaluation**
To train the hyperparameters of the model in the development set, I used the standard segmentation metric window difference. This is a penalty metric between 0 and 1 (0 is the best value) where we slide a window through the reference and hypothesis segmentations and compare the different assignments. This is a more principled way of comparing segmentations because we are taking into account how similar are the "shapes" of two segmentations. This is something that measures such as accuracy do not capture. For example, a segmentation saying that all paragraphs are a single document is better than a segmentation where one boundary was placed very close to an actual document start. The best window difference results I obtained were 0.096774 (in the full development set). The parameters used are in the *configs.json* file of the repository.

## Requirements
- Python 3.6 (recommended)

## Instalation

`pip3 install -r requirements.txt`

## Running

**Document Identification**

Assuming you are in the *PriberamChallenge* directory of the repository use:

`python3 src/doc_id_service.py <any_dataset.json> <prediction.json>`