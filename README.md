# Indeed_Tagging-Raw-Job-Descriptions_HackerRank </br>
Solutions to https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions</br>
## Training dataset quality</br>
Training data set consists of descriptions and corresponding tags </br>
## Data Preprocessing Steps </br>
Creation of Tf-Idf </br>
Creation of response variable </br>
part-time-job or full-time-job [2,1,0] </br>
hourly-wage or salary [2,1,0] </br>
associate-needed [1,0] </br>
bs-degree-needed or ms-or-phd-needed [0-none,1,2] </br>
licence-needed [1,0] </br>
1-year-experience-needed or 2-4-years-experience-needed or 5-plus-years-experience-needed [0-none,1,2,3] </br>
supervising-job [1,0] </br>
nan  [when all of above is 0] </br>
## Explanation and Justification of the Model </br>
Creating multiple models and ensembling them </br>
