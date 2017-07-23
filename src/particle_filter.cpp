#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine random_generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  cout << "particle filter init" << endl;

  num_particles = 12;

  normal_distribution<double> x_normal(x, std[0]);
  normal_distribution<double> y_normal(y, std[1]);
  normal_distribution<double> theta_normal(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
	//for debugging purposes
	//p.x = 0;
	//p.y = 0;
	//p.theta = 0;
    p.x = x_normal(random_generator);
    p.y = y_normal(random_generator);
    p.theta = theta_normal(random_generator);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//cout << "prediction" << endl;
	
	normal_distribution<double> normal_x_prediction(0, std_pos[0]);
    normal_distribution<double> normal_y_prediction(0, std_pos[1]);
    normal_distribution<double> normal_theta_prediction(0, std_pos[2]);
	
	for (int i = 0; i < num_particles; i++) {
		//avoid division by zero
		if (fabs(yaw_rate) <= 0.001) {  
		  particles[i].x += velocity*delta_t*cos(particles[i].theta);
		  particles[i].y += velocity*delta_t*sin(particles[i].theta);
		} else {	
				particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + normal_x_prediction(random_generator);
				particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t))+ normal_y_prediction(random_generator);
				particles[i].theta += yaw_rate*delta_t + normal_theta_prediction(random_generator);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//cout << "data association" << endl;

	//vector for the new associated observations
	std::vector<LandmarkObs> temp;
	
	for (int ii = 0; ii < observations.size() ; ii++) {
		double min_distance = -1;
		int min_index = 0;
		for (int jj = 0; jj < predicted.size() ; jj++) {
			double current_distance = dist(observations[ii].x, observations[ii].y, predicted[jj].x, predicted[jj].y);
			if (min_distance == -1 || current_distance < min_distance){
				min_distance = current_distance;
				min_index = jj;
			}
		}
		temp.push_back(predicted[min_index]);
	}
	observations = temp;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	//cout << "update weights" << endl;
	
	for (int i = 0; i < num_particles; i++) {
		vector<LandmarkObs> predictions;
		
		// limit search to landmarks within range
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
		  double landmark_x = map_landmarks.landmark_list[j].x_f;
		  double landmark_y = map_landmarks.landmark_list[j].y_f;
		  int landmark_id = map_landmarks.landmark_list[j].id_i;
		  
		  if (dist(landmark_x, landmark_y, particles[i].x, particles[i].y) <= sensor_range) {
			LandmarkObs pred_obs;
			pred_obs.id = landmark_id;
			pred_obs.x = landmark_x;
			pred_obs.y = landmark_y;
			predictions.push_back(pred_obs);
		  }
		}
		
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;
		
		//transfor observations to particle coordinates
		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++) {
		  //cout << observations[j] << endl;
		  double transformed_x = observations[j].x*cos(particle_theta)-observations[j].y*sin(particle_theta)+particle_x;
		  double transformed_y = observations[j].x*sin(particle_theta)+observations[j].y*cos(particle_theta)+particle_y;
		  LandmarkObs obs;
		  obs.id = observations[j].id;
		  obs.x = transformed_x;
		  obs.y = transformed_y;
		  //cout << obs << endl;
		  transformed_observations.push_back(obs);
		}
		
		// find associated landmarks
		vector<LandmarkObs> associated_observations = transformed_observations;
		dataAssociation(predictions, associated_observations);
		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < transformed_observations.size(); j++) {
		  
		  double observation_x = transformed_observations[j].x;
		  double observation_y = transformed_observations[j].y;
		  double associated_x = associated_observations[j].x;
		  double associated_y = associated_observations[j].y;

		  //calculate weight
		  double landmark_x = std_landmark[0];
		  double landmark_y = std_landmark[1];
		  double denom = 2*M_PI*landmark_x*landmark_y;
		  double term1 = pow(associated_x-observation_x,2)/(2*pow(landmark_x, 2));
		  double term2 = pow(associated_y-observation_y,2)/(2*pow(landmark_y, 2));
		  double calculated_weight = (1/denom) * exp( -(term1 + term2));

		  particles[i].weight *= calculated_weight;
		}
    }
}

void ParticleFilter::resample() {
	
  //cout << "resample" << endl;

  vector<Particle> new_particles;
  
  weights = *(new std::vector<double>());
  
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> normal(0, num_particles-1);
  int index = normal(random_generator);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> normal_weight(0.0, max_weight);

  double b = 0;

  for (int i = 0; i < num_particles; i++) {
	b += 2*normal_weight(random_generator);
    while (b > weights[index]) {
      b -= weights[index];
      index = (index+1)%num_particles;
	  //cout << b << endl << index << endl;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best){
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best){
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best){
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}