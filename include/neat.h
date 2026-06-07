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

        struct phenotypeConnection
        {
            long int targetNodeID;
            float weight;
        };

        struct phenotypeNeuron
        {
            float bias;
            bool isOutput;
            long int id;
            std::vector<phenotypeConnection> outgoing;
        };

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
            int inputs;
            int outputs;
            float fitness=0;
            float adjustedFitness;
            bool isChampion=false;
            int speciesId=0;
            std::unordered_map<long int, neuronGene> neurons;
            std::vector<long int> neuronList;
            std::vector<synapseGene> synapses; 
            std::unordered_set<long int> synapseIDs;

            std::unordered_map<long int, std::vector<long int>> adj;

            std::vector<phenotypeNeuron> nnExecutionOrder;
        };

        struct pair_hash {
            inline size_t operator()(const std::pair<long int, long int> &v) const {
                return std::hash<long int>{}(v.first) ^ (std::hash<long int>{}(v.second) << 1);
            }
        };

        long int globalInnovationNumber = 0;
        long int globalNeuronNumber = 0;   
        std::random_device rd;  
        std::mt19937 gen; 
        std::uniform_real_distribution<float> distrib;
        std::uniform_real_distribution<float> chance;
        std::vector<genome> genomes;
        std::vector<std::vector<long int>> speciatedGenomes;
        std::vector<std::vector<genome>> newGenomes;
        std::unordered_map<std::pair<long int, long int>, long int, pair_hash> synapseInnovationNumbers;
        std::unordered_map<long int, long int> synapseSplits;
        int populationSize;

        float c1 = 1.0f, c2 = 1.0f, c3 = 0.4f, deltaT = 3.0f, r = 0.2f, disableProb = 0.75f, weightMutationPower = 0.2f;
        float weightMutationRate = 0.80f, biasMutationRate = 0.80f, newSynapseMutationRate = 0.05f, newNeuronMutationRate = 0.03f, flipMutationRate = 0.05f;

        void initialize();
        genome spawnInitial(int inputs, int outputs);
        void generatePopulation(int population, int inputs, int outputs);
        void speciate();
        void fitnessShare();
        genome crossover(genome* parent1, genome* parent2);
        bool hasCycle(long int current, long int target, const std::unordered_map<long int, std::vector<long int>>& graph, std::unordered_set<long int>& visited);
        void mutate(genome* genome); 
        void reproduce(); 
        void compileNetwork(genome& g);
        std::vector<float> feedForward(genome& g, const std::vector<float>& inputs, float (*activationFunction)(float));

        static float sigmoid(float x);
};

#endif