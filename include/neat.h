#ifndef NEAT_H
#define NEAT_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <functional>

class NEAT
{
    public: 
        enum neuronType {INPUT, HIDDEN, OUTPUT};

        struct neuronGene
        {
            long int id;
            float bias;
            neuronType type;
        };

        struct synapseGene
        {
            long int innovationNumber;
            long int inputID;
            long int outputID;
            float weight;
            bool enabled;
        };

        struct genome
        {
            float fitness;
            float adjustedFitness;
            std::unordered_map<long int, neuronGene> neurons;
            std::vector<long int> neuronList;
            std::vector<synapseGene> synapses; 
            std::unordered_set<long int> synapseIDs;

            std::unordered_map<long int, std::vector<long int>> adj;
        };

        struct pair_hash {
            inline size_t operator()(const std::pair<int, int> &v) const {
                return std::hash<int>{}(v.first) ^ (std::hash<int>{}(v.second) << 1);
            }
        };

        long int globalInnovationNumber = 0;
        long int globalNeuronNumber = 0;   
        std::random_device rd;  
        std::mt19937 gen; 
        std::uniform_real_distribution<float> distrib;
        std::uniform_real_distribution<float> chance;
        std::vector<genome> genomes;
        std::vector<std::vector<genome*>> speciatedGenomes;
        std::vector<std::vector<genome>> newGenomes;
        std::unordered_map<std::pair<long int, long int>, int, pair_hash> synapseInnovationNumbers;
        std::unordered_map<long int, long int> synapseSplits;

        float c1 = 1.0f, c2 = 1.0f, c3 = 0.4f, deltaT = 3.0f, r = 0.2f, disableProb = 0.75f, weightMutationPower = 0.2f;

        void initialize();
        genome spawnInitial(int inputs, int outputs);
        void generatePopulation(int population, int inputs, int outputs);
        void speciate();
        void fitnessShare();
        genome crossover(genome* parent1, genome* parent2);
        bool hasCycle(long int current, long int target, const std::unordered_map<long int, std::vector<long int>>& graph, std::unordered_set<long int>& visited);
        void mutate(genome* genome); 
        void reproduce(); 
        std::vector<float> feedForward(genome& g, const std::vector<float>& inputs);
};

#endif