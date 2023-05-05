Download Link: https://assignmentchef.com/product/solved-cse512-hw5-boosting
<br>
We learned about boosting in lecture and the topic is covered in Murphy 16.4. On page 555 Murphy claims that “it was proved that one could boost the performance (on the training set) of any weak learner arbitrarily high, provided the weak learned could always perform slightly better than chance.” We will now verify this in the AdaBoost framework.

<ol>

 <li>(<em>7 points</em>) Given a set of <em>N </em>observations (<em>x<sup>j</sup>,y<sup>j</sup></em>) where <em>y<sup>j </sup></em>is the label <em>y<sup>j </sup></em>∈ {−1<em>,</em>1}, let <em>h<sub>t</sub></em>(<em>x</em>) be the weak classifier at step <em>t </em>and let <em>α<sub>t </sub></em>be its weight. First we note that the final classifier after <em>T </em>steps is defined as:</li>

</ol>

( <em>T                                 </em>)

<em>H</em>(<em>x</em>) = <em>sgn </em><sup>X</sup><em>α<sub>t</sub>h<sub>t</sub></em>(<em>x</em>)              = <em>sgn</em>{<em>f</em>(<em>x</em>)}

<em>t</em>=1

Where

<em>T</em>

<em>f</em>(<em>x</em>) = <sup>X</sup><em>α<sub>t</sub>h<sub>t</sub></em>(<em>x</em>)

<em>t</em>=1 Show that:

Training

Where <em>δ</em>(<em>H</em>(<em>x<sup>j</sup></em>) 6= <em>y<sup>j</sup></em>) is 1 if <em>H</em>(<em>x<sup>j</sup></em>) 6= <em>y<sup>j </sup></em>and 0 otherwise.

<ol start="2">

 <li>(<em>8 points</em>) The weight for each data point <em>j </em>at step <em>t </em>+ 1 can be defined recursively by:</li>

</ol>

Where <em>Z<sub>t </sub></em>is a normalizing constant ensuring the weights sum to 1:

<em>N</em>

<em>Z</em><em>t </em>= X<em>w</em><em>jt </em>exp(−<em>α</em><em>t</em><em>y</em><em>j</em><em>h</em><em>t</em>(<em>x</em><em>j</em>))

<em>j</em>=1

Show that:

<ol start="3">

 <li>(<em>15 points</em>) We showed above that training error is bounded above by. At step <em>t </em>the values <em>Z</em><sub>1</sub><em>,Z</em><sub>2</sub><em>,…,Z<sub>t</sub></em><sub>−1 </sub>are already fixed therefore at step <em>t </em>we can choose <em>α<sub>t </sub></em>to minimize <em>Z<sub>t</sub></em>. Let</li>

</ol>

be the weighted training error for weak classifier <em>h<sub>t</sub></em>(<em>x</em>) then we can re-write the formula for <em>Z<sub>t </sub></em>as:

<ul>

 <li>First find the value of <em>α<sub>t </sub></em>that minimizes <em>Z<sub>t </sub></em>then show that</li>

 <li>Assume we choose <em>Z<sub>t </sub></em>this way. Then re-write where <em>γ<sub>t </sub>&gt; </em>0 implies better than random and <em>γ<sub>t </sub>&lt; </em>0 implies worse than random. Then show that:</li>

</ul>

<em>Z<sub>t </sub></em>≤ exp(−2<em>γ<sub>t</sub></em><sup>2</sup>) You may want to use the fact that log(1 − <em>x</em>) ≤ −<em>x </em>for 0 ≤ <em>x &lt; </em>1

Thus we have:

<em>T                                                  T</em>

training ≤ Y<em>Z</em><em>t </em>≤ exp(−2X<em>γ</em><em>t</em>2)

<em>t</em>=1                                             <em>t</em>=1

<ul>

 <li>Finally, show that if each classifier is better than random (e.g. <em>γ<sub>t </sub></em>≥ <em>γ </em>for all <em>t </em>and <em>γ &gt; </em>0) that:</li>

</ul>

training ≤ exp(−2<em>Tγ</em>2)

Which shows that the training error can be made arbitrarily small with enough steps.

2          Programming Question (clustering with K-means) [30 points]

In class we discussed the K-means clustering algorithm. Your programming assignment this week is to implement the K-means algorithm on digit data.

2.1     The Data

There are two files with the data. The first digit.txt contains the 1000 observations of 157 pixels (a subset of the original 785) from images containing handwritten digits. The second file labels.txt contains the true digit label (either 1, 3, 5, or 7). You can read both data files in with

<ul>

 <li>= load(‘../hw5data/digit/digit.txt’);</li>

 <li>= load(‘../hw5data/digit/labels.txt’);</li>

</ul>

Please note that there aren’t IDs for the digits. Please assume the first line is ID 1, the second line is ID 2, and so on. The labels correspond to the digit file, so the first line of labels.txt is the label for the digit in the first line of digit.txt.

2.2     The algorithm

Your algorithm should be implemented as follows:

<ol>

 <li>Select <em>k </em>starting centers that are points from your data set. You should be able to select these centers randomly or have them given as a parameter.</li>

 <li>Assign each data point to the cluster associated with the nearest of the <em>k </em>center points.</li>

 <li>Re-calculate the centers as the mean vector of each cluster from (2).</li>

 <li>Repeat steps (2) and (3) until convergence or iteration limit.</li>

</ol>

Define convergence as no change in label assignment from one step to another or you have iterated 20 times (whichever comes first). Please count your iterations as follows: after 20 iterations, you should have assigned the points 20 times.

<ul>

 <li>Within group sum of squares</li>

</ul>

The goal of clustering can be thought of as minimizing the variation within groups and consequently maximizing the variation between groups. A good model has low sum of squares within each group.

We define sum of squares in the traditional way. Let C<em><sub>k </sub></em>be the <em>k<sup>th </sup></em>cluster and let <em>µ</em><em><sub>k </sub></em>be the mean of the observations in cluster C<em><sub>k</sub></em>. Then the within group sum of squares for cluster C<em><sub>k </sub></em>is defined as:

<em>SS</em>(<em>k</em>) = <sup>X </sup>||<strong>x</strong><em><sub>i </sub></em>− <em>µ</em><em><sub>k</sub></em>||<sup>2</sup>

<em>i</em>∈C<em><sub>k</sub></em>

Please note that the term ||<strong>x</strong><em><sub>i </sub></em>− <em>µ</em><em><sub>k</sub></em>||<sup>2 </sup>is the euclidean distance between <strong>x</strong><em><sub>i </sub></em>and <em>µ</em><em><sub>k</sub></em>.

If there are <em>K </em>clusters total then the “total within group sum of squares” is just the sum of all <em>K </em>of these individual <em>SS</em>(<em>k</em>) terms.

<ul>

 <li>Pair-counting measures</li>

</ul>

Given that we know the actual assignment labels for each data point we can attempt to analyze how well the clustering recovered this. WE will performance criteria which are based on two principles: i) two data points that belong to the same class should be assigned to the same cluster; and ii) two data points that belong to different classes should be assigned to different clusters. Formally speaking, consider all pairs of same-class data points, let <em>p</em><sub>1 </sub>be the percentage of pairs of which both data points were assigned to the same cluster. Consider all pairs of different-class data points, let <em>p</em><sub>2 </sub>be the percentage of pairs of which two data points are assigned to different clusters. Let <em>p</em><sub>3 </sub>be the average of these two values <em>p</em><sub>3 </sub>= (<em>p</em><sub>1</sub>+<em>p</em><sub>2</sub>)<em>/</em>2, which summarizes the clustering performance.

<ul>

 <li>Questions</li>

</ul>

When you have implemented the algorithm please report the following:

<ol>

 <li>[8pts] The values of the total within group sum of squares and pair-counting measures (<em>p</em><sub>1</sub><em>,p</em><sub>2</sub><em>,p</em><sub>3</sub>) for <em>k </em>= 2, <em>k </em>= 4 and <em>k </em>= 6. Start your centers with the first <em>k </em>points in the dataset. So, if <em>k </em>= 2, your initial centroids will be ID 1 and ID 2, which correspond to the first two lines in the file.</li>

 <li>[7pts] The number of iterations that k-means ran for <em>k </em>= 6, starting the centers as in the previous item. Make sure you count the iterations correctly. If you start with iteration <em>i </em>= 1 and at <em>i </em>= 4 the cluster assignments don’t change, the number of iterations was 4, as you had to do step 2 four times to figure this out.</li>

 <li>[7pts] A plot of the total within group sum of squares versus <em>k </em>for <em>k </em>= 1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5<em>,</em>6<em>,</em>7<em>,</em>8<em>,</em>9<em>,</em>10. Start your centers randomly (choose <em>k </em>points from the dataset at random).</li>

 <li>[8pts] A plot of <em>p</em><sub>1</sub><em>,p</em><sub>2</sub><em>,p</em><sub>3 </sub>versus <em>k </em>for <em>k </em>= 1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5<em>,</em>6<em>,</em>7<em>,</em>8<em>,</em>9<em>,</em>10. Start your centers randomly (choose <em>k </em>points from the dataset at random).</li>

</ol>

For the last two items, you should run <em>k</em>-means algorithm several times (e.g., 10 times) and average the results. For each question, submit a single plot, which is the average of the runs.

3         Programming Question (scene classification)

This question gives you the opportunities to learn LibSVM, which is one of the best software tool for classification problem. You will train SVMs with different kernels and use them to classify scenes from The Big Bang Theory, your favorite TV series. To classify an image, you will use Bag-of-Word representation, which is one of the most popular image representation. This representation views an images as the histogram of image features, or visual words. You MUST use LibSVM and your K-means implementation from Question 3.

3.1      LibSVM installation

First download LibSVM <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">https://www.csie.ntu.edu.tw/</a><a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">˜</a><a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">cjlin/libsvm/</a><a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">.</a> Follow the instruction in README to install LibSVM for Matlab or Python. Two main functions of LibSVM that you should pay attention to are: svmtrain and svmpredict. Note that Matlab also has a machine learning toolbox that comes with these two functions with exactly the same names. However, Matlab’s SVM implementation is not as good as LibSVM, so you need to make sure that you are using svmtrain and svmpredict from LibSVM. To check if you have installed the program correctly, in Matlab do:

&gt;&gt; which svmtrain

&gt;&gt; which svmpredict

Matlab should return the paths to the svmtrain and svmpredict of LibSVM. To learn how to use these functions, type the names of the function in Matlab:

&gt;&gt; svmtrain

&gt;&gt; svmpredict

3.2     Data

Training and test images are provided in the subdirectory bigbangtheory. The training image ids and labels are given in train.mat. This file contains two variables: imgIds and lbs. imgIds is a column vector and each row has a name of image in the training set. lbs is a matrix denoting the label for the image with the corresponding index. There are total 8 classes for the dataset: living room (1), kitchen (2), hallway (3), Penny’s living room (4), cafeteria (5), Cheesecake factory (6), laundry room (7), and comic bookstore (8).

Validation set is not provided for this question. You have to do cross validation to find the parameter for the best performance. You can implement cross validation by yourself, or you can use LibSVM functionality. Image ids for test set are given in test.mat.

3.3     Helper functions

A number of Matlab helper functions are provided. Also, some functions are left unfinished and you have to complete them.

<ol>

 <li>Use HW5BoW.learnDictionary() to learn visual vocabulary for scene representation. You have to fill your K-means implementation from Question 3 in this function.</li>

 <li>Use HW5BoW.cmpFeatVecs() to compute feature vectors. This function will return the histogram of visual words for each image, which is our feature vector.</li>

</ol>

3.4     What to implement?

<ol>

 <li>Complete the function HW5BoW.learnDictionary(). This function learns a visual vocabulary by running <em>k</em>-means on random image patches. Learn a visual dictionary with <em>K </em>= 1000</li>

 <li>(10 points) Using SVM with RBF kernel, report the 5-fold cross-validation accuracy for the default kernel parameters for <em>C </em>and <em>γ</em>.</li>

 <li>(10 points) Tune <em>C </em>and <em>γ </em>for SVM with RBF kernel using 5-fold cross validation. Report the values of <em>C</em>, <em>γ</em>, and the 5-fold cross-validation accuracy.</li>

 <li>Unfortunately, LibSVM does not support exponential X<sup>2 </sup>kernel, so you will need to implement it.</li>

</ol>

Implement a function:

[trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)

that takes train and test data and the kernel parameter gamma and return the train and test kernels. Recall that the exponential X<sup>2 </sup>kernel value for two d-dimensional feature vector <strong>x </strong>and <strong>y </strong>is:

!

(1)

The <em> </em>term is added to avoid division by 0.

<ol start="5">

 <li>(10 points) LibSVM allows using pre-computed kernels. Train an SVM with exponential X<sup>2 </sup> Report the 5-fold cross validation accuracy with the best tuned values of <em>C </em>and <em>γ</em>. Note that a good default value of <em>γ </em>is the average of X<sup>2 </sup>distance between training data points, as discussed in the lecture.</li>

 <li>(10 points) Use your model to predict on the test set. Save the predicted labels on a CSV file predTestLabels.csv (see Section 3 for the format). The order of predicted labels should correspond to the order of the reviews in test.mat. Submit this predTestLabels.csv file on Kaggle and report classification accuracy in your answers.pdf file. You must use Stony Brook NetID email to register for Kaggle.</li>

</ol>

Note: Marks for this question will be scaled according to the ranking on the Private Leaderboard.

<ol start="7">

 <li>(10 bonus points) We will maintain a leader board on Kaggle, and the top three entries at the end of the competition (due date) will receive 10 bonus points. You must use your real names and Stony Brook Net ID on Kaggle to compete for the bonus points.</li>

</ol>

To prevent exploiting test data, you are allowed to make a maximum of 3 submissions per 24 hours. Your submission will be evaluated immediately and the leader board will be updated.