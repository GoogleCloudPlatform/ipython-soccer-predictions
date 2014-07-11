ipython-soccer-predictions
==========================

## Machine learning applied to soccer

Sample IPython notebook with soccer predictions

We’ve had a great time giving you our predictions for the World Cup (check out our [post before the quarter finals](http://googlecloudplatform.blogspot.com/2014/07/google-cloud-platform-goes-8-for-8-in-soccer-predictions.html) and the one [before the semi-finals](http://googlecloudplatform.blogspot.com/2014/07/google-cloud-platform-is-11-for-12-in-World-Cup-predictions.html)). So far, we’ve gotten 13 of 14 games correct. But this shouldn’t be about what we did - it’s about what you can do with Google Cloud Platform. Now, we are open-sourcing our prediction model and packaging it up so you can do your own analysis and predictions. 

We have ingested raw touch-by-touch gameplay day from Opta for thousands of soccer matches using Google Cloud Dataflow and polished the raw data into predictive statistics using Google BigQuery. You can see BigQuery engineer Jordan Tigani (+JordanTigani) and developer advocate Felipe Hoffa (@felipehoffa) talk about how we did it in this [video from Google I/O](https://www.youtube.com/watch?v=YyvvxFeADh8). 


## Project setup, installation, and configuration


## Deploying

### How to setup the deployment environment

Pre-work: Get started with the Google Cloud Platform and create a project:

Sign up at https://console.developers.google.com/, create a project, and remember to turn on the Google BigQuery API. Install the Google Cloud SDK following the instructions at https://developers.google.com/cloud/sdk/.


### How to deploy

Start your instance:

`gcloud compute instances create ipy-predict --image https://www.googleapis.com/compute/v1/projects/google-containers/global/images/container-vm-v20140522 --zone=us-central1-a --machine-type n1-standard-1 --scopes storage-ro bigquery`

Ssh to your new machine:

`gcutil ssh --ssh_arg "-L 8888:127.0.0.1:8888" --zone=us-central1-a ipy-predict`


Download and run the docker image we prepared:

`sudo docker run -p 8888:8888 fhoffa/ipython-predictions:v1`

Wait until Docker downloads and runs the container, then navigate to the notebook:

http://127.0.0.1:8888/notebooks/worldcup/regression/wc-final.ipynb


## Contributing changes

* See [CONTRIB.md](CONTRIB.md)


## Licensing

* See [LICENSE](LICENSE)
