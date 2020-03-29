# COVID 19 Detection Using Deep Learning
##### Using Deep Learning to detect COVID-19 from Chest X-rays. It can be used as an early pre-screening for COVID-19 by hospitals, because virology tests for the infection take a long time to conduct, especially due to hospitals being overwhelmed with so many cases.
&nbsp;

## Description
It is simple. Just upload a JPG/PNG image format of your X-Ray or CT Scan and click Submit!
Our model tells you whether you have tested positive or negative for COVID-19 along with the prediction score associated with it.

## Inspiration
The real time reverse transcription-polymerase chain reaction (RT-PCR) detection of viral RNA from sputum or nasopharyngeal swab has a relatively low positive rate in the early stage to determine COVID-19. Due to this, an infected person might test negative in the earlier stages even if he/she is carrying the infection in their system. The manifestations of X-Ray imaging and computed tomography (CT) of COVID-19 have their own characteristics according to recent research and observations published online, which are different from other types of viral pneumonia, such as Influenza-A viral pneumonia. Therefore, the need of the hour calls for an early diagnostic criteria for this new type of infection as soon as possible. This web application is aimed to establish an early screening model to distinguish COVID-19 cases from other Influenza-A viral pneumonia and healthy cases with X-Ray and CT images using deep learning techniques.
A person who gets tested positive for the disease using our system can take an immediate initiative to self-quarantine himself/herself and avoid the infection from spreading to other people since the virology tests can take a significant amount of time to give concrete results, especially due to the hospitals and health facilities of USA being overwhelmed with a large number of COVID-19 cases.

### Technology

This project uses a number of open source projects to work properly:

* [Python] - awesome language we love
* [Tensorflow] - An end-to-end open source machine learning platform for everyone
* [Teachable Machine by Google] : To train a computer to recognize your own images, sounds, & poses.

#### Installation

This project requires the standard [Python](https://www.python.org/) 3.6+ to run

```sh
$ git clone https://github.com/SukhyAnand/COVID-19-Detection-Using-Deep-Learning.git
$ cd COVID-19-Detection-Using-Deep-Learning
$ pip install -r requirements.txt
```

#### Executing the app

```
$ python app.py
```
