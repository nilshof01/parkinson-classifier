Overview
Parkinsonian syndromes affect millions of people worldwide, yet diagnosing them accurately and early remains challenging. Dopamine transporter (DaT) imaging is an important tool for distinguishing neurodegenerative parkinsonian syndromes from other conditions, but reliable interpretation requires specialized expertise that is not available everywhere.

Each year, more than 20,000 DaT scans are performed in France alone. While many can be classified as normal or abnormal, approximately one in five cases remains difficult to interpret — particularly in early-stage or atypical presentations — delaying diagnosis, complicating treatment decisions, and increasing demands on specialist readers.

Task
In this challenge, we invite data scientists, machine learning engineers, medical imaging researchers, and nuclear medicine specialists to advance AI tools for DaT scan interpretation. Using a unique multicenter dataset of scans collected and annotated by French experts across ten hospital centers in France, your goal is to develop computer vision models that classify DaT scans as normal or abnormal.

Successful models will be released as open source and indexed in the the Bibliothèque Ouverte d'Algorithmes en Santé (BOAS), France's open health algorithm library, while the dataset will be made available as open data on data.gouv. By improving the accuracy and consistency of DaT scan interpretation, winning solutions could help expand access to expert-level diagnostic support and accelerate research into neurodegenerative disorders.

Prizes
Competition End Date:
Sept. 16, 2026, 11:59 p.m. UTC
Place	Prize Amount
1st	€12,500
2nd	€7,500
3rd	€5,000
Total	€25,000
Note: Prizes delivered by DrivenData in USD, based on the exchange rate on July 22, 2026.

After the challenge
Updated July 29, 2026 with additional details on post-challenge events for competition winners

Winners of the competition will be invited to an in-person ceremony in France, hosted by the French Society of Nuclear Medicine (SFMN) and France's Health Data Hub, where they'll present their solutions and be recognized for their achievements.

Top teams may also be contacted after the challenge concludes about an optional scientific collaboration with SFMN, exploring age and sex prediction from DaT scans.

How to compete
Click the "Compete!" button in the sidebar to enroll in the competition.
Get familiar with the problem through the problem description. Additional resources are available on the about page.
Download the data from the data download page.
Create and train your own model.
Package your model files with the code to make predictions based on the runtime repository specification on the code submission format page.
Test your submission locally using the instructions in the runtime repository, and in the smoke test environment.
From the submissions page, submit your code as a ZIP archive for containerized execution. You're in!
Competition rules
Below are a few highlights of the rules. See the full competition rules for complete details. They are designed to promote fair competition and encourage useful, reproducible solutions. If you are ever unsure whether your solution complies with the rules, ask in the competition forum or contact the organizers.

Competition data
Participants must:

Use data only for this challenge and only during the challenge period.
Delete all local data after the competition ends, unless a separate license allows continued use.
Never share, copy, or publish the data. For example, you may not use tools like Codex and ChatGPT that store or retain uploaded data, though you may download model weights and run models locally.
External data and models
External data and pre-trained models are allowed in this competition. Participants may use external data provided they have the legal right to do so. All external data must be shared with the challenge organizers, regardless of prize eligibility, to allow for independent result verification. Additionally, participants may not use tools like Codex or ChatGPT that store or retain uploaded data, but may download model weights and run models locally.

This challenge aims to support open solutions with broad social benefit and real-world applicability. To be eligible for prizes, (1) any external data must be freely and publicly available to all participants; and (2) any external data or pre-trained models used must be licensed so that the resulting model can be released for broad use, in and beyond the competition, including for commercial purposes (no NC, CC NC, or CC BY-NC licenses).

If you have questions about licensing in general or whether specific external data can be used, post in the competition forum or send an email to info@drivendata.org.

Organized by SFMN
This challenge is organized by the French Society of Nuclear Medicine (SFMN), in partnership with the Health Data Hub and GaelO. It is part of the "Health Data Challenges" call for projects, financially supported by the France 2030 plan. The competition dataset was assembled through a nationwide, multicenter collaboration of hospitals across France — learn more on the about page.


Problem description
Your goal is to develop machine learning models that predict the probability that a dopamine transporter (DaT) scan examination is abnormal rather than normal.

In this challenge, you will build models that distinguish normal from abnormal DaT scans using a large multicenter dataset collected and annotated by French experts. Successful solutions should generalize across institutions, scanners, and patient populations, and could help improve the consistency, accessibility, and scalability of DaT scan interpretation.

Dataset
The imaging data are provided as three-dimensional DaT scan reconstructions in compressed Neuroimaging Informatics Technology Initiative (NIfTI) format (.nii.gz). Each examination is a single 3D volume, and each patient is represented by one file. The filename (minus the .nii.gz extension) is the uid that links each image to its label.

In addition, participants are welcome to incorporate external datasets for training; however, use of external data is subject to important exceptions and caveats (see External Data).

Participants are not allowed to share the competition data or use it for any purpose other than this competition. Participants cannot send the data to any third-party service or API, including, but not limited to, OpenAI's ChatGPT, Google's Gemini, or similar tools. For complete details, review the competition rules.

Files
data/
├── niftis/
│   ├── xaji0y6d.nii.gz
│   ├── pbhsahxt.nii.gz
│   └── ...
└── train_labels.csv
Images
Each .nii.gz file contains a single 3D reconstructed volume stored as 16-bit unsigned integer voxel intensities. Note that acquisition and reconstruction parameters vary across the dataset, so images are not all the same size or resolution:

Volume dimensions differ from scan to scan (for example, 142 × 142 × 112 or 128 × 128 × 128). Your pipeline should not assume a fixed input shape.
Voxel spacing also varies (e.g., 2.46 mm and 3.895 mm isotropic), reflecting differences in scanners and acquisition protocols across contributing centers. The spacing is recorded in each file's NIfTI header and may be useful for resampling images to a common resolution.
These differences are a normal consequence of pooling data from multiple institutions, and building models that generalize across them is part of the challenge.

Image example
Here's an example DaT scan examination, shown as several axial slices moving through the volume:

Several axial slices through an example DaT scan, displayed with a hot colormap

Note: Each file is a single 3D reconstruction, and dopamine transporter uptake is concentrated in the striatum near the center of the brain. Examinations vary in size and voxel spacing across scanners and centers, so the number and appearance of slices will differ from scan to scan.
Labels
train_labels.csv contains one row per DaT scan examination in the training set, with the following columns:

uid (str) — unique identifier for each DaT scan examination; matches the image filename without the .nii.gz extension
is_pathologic (float) — classification of the DaT scan examination, where 0.0 = Normal and 1.0 = abnormal
Label example — the first five rows of train_labels.csv:

uid	is_pathologic
xaji0y6d	0.0
pbhsahxt	0.0
hv3a3zmf	0.0
8mdd4v30	0.0
t9nt3w5u	1.0
Test set
The test set is withheld and is not available for download. Its images are only accessible from within the runtime container, where they are mounted alongside the training data. Because this is a code execution challenge, you will not see the test examinations directly — your submitted code reads them at inference time and generates predictions for each one.

The test examinations are the same NIfTI format as the training data and follow the same conventions (one 3D volume per file, named <uid>.nii.gz, with varying dimensions and voxel spacing).

Performance metric
Leaderboard performance is evaluated according to log loss. Log loss (a.k.a. logistic loss or cross-entropy loss) penalizes confident but incorrect predictions. It also rewards confidence scores that are well-calibrated probabilities, meaning that they accurately reflect the long-run probability of being correct. This is an error metric, so a lower value is better.

Log loss for a single observation is calculated as follows:

𝐿log⁡(𝑦,𝑝)=−(𝑦⁢log⁡(𝑝)+(1−𝑦)⁢log⁡(1−𝑝))

where 𝑦 is a binary variable indicating whether the examination is abnormal (1) or normal (0), and 𝑝 is the user-predicted probability that the examination is abnormal. The loss for the entire dataset is the average loss across all observations.

Note: log loss can often be improved with calibration. A well-calibrated model outputs predictions that are directly interpretable as probabilities.

The leaderboard also displays Area Under the Receiver Operating Characteristic (AUROC) for reference, but this metric does not affect leaderboard ranking or prizes.

Final prizes will be awarded based on the private leaderboard ranking, which is determined using a private test set that may be different from the test set underlying the public leaderboard.

Submission format
This is a code execution challenge.

Rather than submitting predictions directly, participants will package their trained model and inference code for containerized execution.

Your code must read the test examinations and produce a submission.csv with one row per examination:

uid (str) — unique identifier for each DaT scan examination
is_pathologic (float) — your predicted probability that the examination is abnormal, as a value between 0 and 1
Because performance is evaluated with log loss, submit calibrated probabilities rather than discrete 0/1 labels.

Additional information about the runtime environment, package structure, and submission requirements can be found on the code submission format page.