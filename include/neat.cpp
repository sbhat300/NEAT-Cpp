#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <queue>

#include "neat.h"

void NEAT::initialize()
{
    gen = std::mt19937(rd());
    distrib = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    chance = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

NEAT::genome NEAT::spawnInitial(int inputs, int outputs)
{
    genome newGenome;
    newGenome.inputs = inputs;
    newGenome.outputs = outputs;
    long int globalInnovationCounter = 0;
    for(int i = 0; i < inputs; i++)
    {
        newGenome.neurons[i] = {i, (float)distrib(gen), INPUT};
        newGenome.neuronList.push_back(i);
    }
    for(int i = inputs; i < inputs + outputs; i++)
    {
        newGenome.neurons[i] = {i, (float)distrib(gen), OUTPUT};
        newGenome.neuronList.push_back(i);
    }
    for(int i = 0; i < inputs; i++)
    {
        for(int j = 0; j < outputs; j++)
        {
            newGenome.synapses.push_back({globalInnovationCounter, i, j + inputs, (float)distrib(gen), true});
            synapseInnovationNumbers[{i, j + inputs}] = globalInnovationCounter;
            newGenome.synapseIDs.insert(globalInnovationCounter);
            newGenome.adj[i].push_back(j + inputs);
            globalInnovationCounter++;
        }
    }
    if(globalInnovationNumber < globalInnovationCounter) globalInnovationNumber = globalInnovationCounter;
    return newGenome;
}

void NEAT::generatePopulation(int population, int inputs, int outputs)
{
    globalNeuronNumber = inputs + outputs;
    populationSize = population;
    for(int i = 0; i < population; i++)
    {
        genomes.push_back(spawnInitial(inputs, outputs));
    }
}

void NEAT::speciate()
{
    if(genomes.empty()) return;
    speciatedGenomes.clear();
    speciatedGenomes.push_back(std::vector<long int>());
    speciatedGenomes[0].push_back(0);

    for(int i = 1; i < genomes.size(); i++)
    {
        bool found = false;
        for(long int j = 0; j < speciatedGenomes.size(); j++)
        {
            genome* compare = &genomes[speciatedGenomes[j][0]];
            long int excess = 0, disjoint = 0, matching = 0;
            long int max1 = 0, max2 = 0;
            float weightDiff = 0;

            long int p1 = 0, p2 = 0;
            while(p1 < genomes[i].synapses.size() || p2 < compare->synapses.size())
            {
                if(p1 == genomes[i].synapses.size())
                {
                    excess += compare->synapses.size() - p2;
                    break;
                }
                if(p2 == compare->synapses.size())
                {
                    excess += genomes[i].synapses.size() - p1;
                    break;
                }

                long int id1 = genomes[i].synapses[p1].innovationNumber, id2 = compare->synapses[p2].innovationNumber;
                if(id1 == id2)
                {
                    matching++;
                    weightDiff += std::abs(genomes[i].synapses[p1].weight - compare->synapses[p2].weight);
                    p1++;
                    p2++;
                }
                else if(id1 < id2)
                {
                    disjoint++;
                    p1++;
                }
                else
                {
                    disjoint++;
                    p2++;
                }
            }

            float W = 0;
            if(matching != 0) W = weightDiff / matching;
            long int N = std::max((long int)std::max(genomes[i].synapses.size(), compare->synapses.size()), (long int)1);
            float delta = c1 * (float)excess / N + c2 * (float)disjoint / N + c3 * W;
            if(delta <= deltaT) 
            {
                speciatedGenomes[j].push_back(i);
                found = true;
                break;
            }
        }
        if(!found)
        {
            speciatedGenomes.push_back(std::vector<long int>());
            speciatedGenomes[speciatedGenomes.size() - 1].push_back(i);
        }
    }
}

void NEAT::fitnessShare()
{
    float meanAdjFitness = 0;
    std::vector<float> NPrime(speciatedGenomes.size());

    float maxFitness = -1.0f;
    int bestSpecies = 0;
    
    for(int i = 0; i < speciatedGenomes.size(); i++)
    {
        NPrime[i] = 0;
        for(int j : speciatedGenomes[i])
        {
            genome* g = &genomes[j];
            float adjustedFitness = g->fitness / speciatedGenomes[i].size();
            NPrime[i] += adjustedFitness;
            meanAdjFitness += adjustedFitness;

        }
        if(NPrime[i] > maxFitness)
        {
            maxFitness = NPrime[i];
            bestSpecies = i;
        }
    }
    meanAdjFitness /= genomes.size();
    
    newGenomes.clear();
    newGenomes.resize(speciatedGenomes.size());
    int allocated = 0;
    if(meanAdjFitness == 0) 
    {
        int perSpecies = genomes.size() / speciatedGenomes.size();
        int remainder = genomes.size() % speciatedGenomes.size();
        for(int i = 0; i < speciatedGenomes.size(); i++)
        {
            newGenomes[i].resize(perSpecies);
        }
        newGenomes[0].resize(perSpecies + remainder);
        return;
    }

    for(int i = 0; i < speciatedGenomes.size(); i++) 
    {
        long int allocate = (long int)(NPrime[i] / meanAdjFitness);
        newGenomes[i].resize(allocate);
        allocated += allocate;
    }
    int remainder = genomes.size() - allocated;
    newGenomes[bestSpecies].resize(newGenomes[bestSpecies].size() + remainder);
}

NEAT::genome NEAT::crossover(genome* parent1, genome* parent2)
{
    genome newGenome;
    newGenome.inputs = parent1->inputs;
    newGenome.outputs = parent2->outputs;
    genome* strongParent;
    genome* weakParent;
    if(parent1->fitness > parent2->fitness)
    {
        strongParent = parent1;
        weakParent = parent2;
    }
    else
    {
        strongParent = parent2;
        weakParent = parent1;
    }

    if(parent1->fitness == parent2->fitness)
    {
        std::unordered_set<int> pushedNeurons;
        for(std::pair<const long int, neuronGene>& neuron : parent1->neurons) 
        {
            pushedNeurons.insert(neuron.first);
            newGenome.neurons[neuron.first] = neuron.second;
            newGenome.neuronList.push_back(neuron.first);
        }
        for(std::pair<const long int, neuronGene>& neuron : parent2->neurons) 
        {
            if(pushedNeurons.find(neuron.first) == pushedNeurons.end()) 
            {
                newGenome.neurons[neuron.first] = neuron.second;
                newGenome.neuronList.push_back(neuron.first);
            }
        }
    } 
    else
        for(std::pair< const long int, neuronGene>& neuron : strongParent->neurons) 
        {
            newGenome.neurons[neuron.first] = neuron.second;
            newGenome.neuronList.push_back(neuron.first);
        }

    long int p1 = 0, p2 = 0;
    while(p1 < strongParent->synapses.size() || p2 < weakParent->synapses.size())
    {
        if(p1 == strongParent->synapses.size())
        {
            if(strongParent->fitness == weakParent->fitness)
            {
                while(p2 < weakParent->synapses.size()) 
                {
                    newGenome.synapses.push_back(weakParent->synapses[p2]);
                    newGenome.synapseIDs.insert(weakParent->synapses[p2].innovationNumber);
                    newGenome.adj[weakParent->synapses[p2].inputID].push_back(weakParent->synapses[p2].outputID); 
                    p2++;
                }
            }
            break;
        }
        if(p2 == weakParent->synapses.size())
        {
            while(p1 < strongParent->synapses.size()) 
            {
                newGenome.synapses.push_back(strongParent->synapses[p1]);
                newGenome.synapseIDs.insert(strongParent->synapses[p1].innovationNumber);
                newGenome.adj[strongParent->synapses[p1].inputID].push_back(strongParent->synapses[p1].outputID); 
                p1++;
            }
            break;
        }

        long int id1 = strongParent->synapses[p1].innovationNumber, id2 = weakParent->synapses[p2].innovationNumber;
        if(id1 == id2)
        {
            if(chance(gen) < 0.5f) 
            {
                newGenome.synapses.push_back(strongParent->synapses[p1]);
                newGenome.synapseIDs.insert(strongParent->synapses[p1].innovationNumber);
                newGenome.adj[strongParent->synapses[p1].inputID].push_back(strongParent->synapses[p1].outputID); 
            }
            else 
            {
                newGenome.synapses.push_back(weakParent->synapses[p2]);
                newGenome.synapseIDs.insert(weakParent->synapses[p2].innovationNumber);
                newGenome.adj[weakParent->synapses[p2].inputID].push_back(weakParent->synapses[p2].outputID); 
            }
            if(strongParent->synapses[p1].enabled && weakParent->synapses[p2].enabled) newGenome.synapses.back().enabled = true;
            else
            {
                if(chance(gen) < disableProb) newGenome.synapses.back().enabled = false;
                else newGenome.synapses.back().enabled = true;
            }
            p1++;
            p2++;
        }
        else if(id1 < id2)
        {
            newGenome.synapses.push_back(strongParent->synapses[p1]);
            newGenome.synapseIDs.insert(strongParent->synapses[p1].innovationNumber);
            newGenome.adj[strongParent->synapses[p1].inputID].push_back(strongParent->synapses[p1].outputID); 
            p1++;
        }
        else
        {
            if(parent1->fitness == parent2->fitness) 
            {
                newGenome.synapses.push_back(weakParent->synapses[p2]);
                newGenome.synapseIDs.insert(weakParent->synapses[p2].innovationNumber);
                newGenome.adj[weakParent->synapses[p2].inputID].push_back(weakParent->synapses[p2].outputID); 
            }
            p2++;
        }
    }
    return newGenome;
}

bool NEAT::hasCycle(long int current, long int target, const std::unordered_map<long int, std::vector<long int>>& graph, std::unordered_set<long int>& visited)
{
    if(current == target) return true;
    if (visited.find(current) != visited.end()) return false;
    visited.insert(current);
    auto it = graph.find(current);
    if(it != graph.end())
    {
        for(long int next : it->second) 
            if(hasCycle(next, target, graph, visited)) return true;
    }
    return false;
}

void NEAT::mutate(genome* genome)
{
    for(synapseGene& synapse : genome->synapses)
    {
        if(chance(gen) < weightMutationRate) 
        {
            if(chance(gen) < 0.90f) synapse.weight += distrib(gen) * weightMutationPower;
            else synapse.weight = distrib(gen);
        }
    }
    for(std::pair<const long int, neuronGene>& neuron : genome->neurons)
    {
        if(chance(gen) < biasMutationRate) 
        {
            if(chance(gen) < 0.90f) neuron.second.bias += distrib(gen) * weightMutationPower;
            else neuron.second.bias = distrib(gen);
        }
    }

    if(chance(gen) < newSynapseMutationRate)
    {
        std::uniform_int_distribution<long int> neuronDist(0, genome->neuronList.size() - 1);
        neuronGene* first = &genome->neurons[genome->neuronList[neuronDist(gen)]];
        neuronGene* second = &genome->neurons[genome->neuronList[neuronDist(gen)]];
        if(first->id != second->id && second->type != INPUT)
        {
            synapseGene newSynapse = {globalInnovationNumber, first->id, second->id, (float)distrib(gen), true};
            bool synapseExists = synapseInnovationNumbers.find({first->id, second->id}) != synapseInnovationNumbers.end();
            if(synapseExists) newSynapse.innovationNumber = synapseInnovationNumbers[{first->id, second->id}];
            else newSynapse.innovationNumber = globalInnovationNumber;
            if(genome->synapseIDs.find(newSynapse.innovationNumber) == genome->synapseIDs.end())
            {
                std::unordered_set<long int> visited;

                if(!hasCycle(second->id, first->id, genome->adj, visited))
                {
                    if(!synapseExists)synapseInnovationNumbers[{newSynapse.inputID, newSynapse.outputID}] = globalInnovationNumber++;
                    int len = genome->synapses.size();
                    genome->synapseIDs.insert(newSynapse.innovationNumber);
                    auto it = std::upper_bound(genome->synapses.begin(), genome->synapses.end(), newSynapse, [](const synapseGene& a, const synapseGene& b) {
                        return a.innovationNumber < b.innovationNumber;
                    });
                    genome->synapses.insert(it, newSynapse);
                    genome->adj[first->id].push_back(second->id);

                }
            }
        } 

    }

    if(chance(gen) < newNeuronMutationRate)
    {
        std::vector<int> enabledIndices;
        enabledIndices.reserve(genome->synapses.size());

        for(int i = 0; i < genome->synapses.size(); i++)
        {
            if(genome->synapses[i].enabled) enabledIndices.push_back(i);
        }
        if(!enabledIndices.empty())
        {
            std::uniform_int_distribution<long int> enabledDist(0, enabledIndices.size() - 1);
            synapseGene splitSynapse = genome->synapses[enabledDist(gen)];
            long int newNeuronID;
            bool neuronDoesNotExist = synapseSplits.find(splitSynapse.innovationNumber) == synapseSplits.end();
            if(neuronDoesNotExist) newNeuronID = globalNeuronNumber++;
            else newNeuronID = synapseSplits[splitSynapse.innovationNumber];

            genome->neurons[newNeuronID] = {newNeuronID, (float)distrib(gen), HIDDEN};
            genome->neuronList.push_back(newNeuronID);
            if(neuronDoesNotExist) synapseSplits[splitSynapse.innovationNumber] = newNeuronID;

            splitSynapse.enabled = false;

            int synapseID;
            bool splitExists = synapseInnovationNumbers.find({splitSynapse.inputID, newNeuronID}) != synapseInnovationNumbers.end();
            if(splitExists) synapseID = synapseInnovationNumbers[{splitSynapse.inputID, newNeuronID}];
            else synapseID = globalInnovationNumber++;
            synapseGene newSynapse = {synapseID, splitSynapse.inputID, newNeuronID, 1.0f, true};
            genome->synapseIDs.insert(synapseID);
            if(!splitExists) synapseInnovationNumbers[{splitSynapse.inputID, newNeuronID}] = synapseID;
            auto it = std::upper_bound(genome->synapses.begin(), genome->synapses.end(), newSynapse, [](const synapseGene& a, const synapseGene& b) {
                return a.innovationNumber < b.innovationNumber;
            });
            genome->synapses.insert(it, newSynapse);
            genome->adj[splitSynapse.inputID].push_back(newNeuronID);
            
            splitExists = synapseInnovationNumbers.find({newNeuronID, splitSynapse.outputID}) != synapseInnovationNumbers.end();
            if(splitExists) synapseID = synapseInnovationNumbers[{newNeuronID, splitSynapse.outputID}];
            else synapseID = globalInnovationNumber++;
            newSynapse = {synapseID, newNeuronID, splitSynapse.outputID, splitSynapse.weight, true};
            genome->synapseIDs.insert(synapseID);
            if(!splitExists) synapseInnovationNumbers[{newNeuronID, splitSynapse.outputID}] = synapseID;
            it = std::upper_bound(genome->synapses.begin(), genome->synapses.end(), newSynapse, [](const synapseGene& a, const synapseGene& b) {
                return a.innovationNumber < b.innovationNumber;
            });
            genome->synapses.insert(it, newSynapse);
            genome->adj[newNeuronID].push_back(splitSynapse.outputID);
        }
    }

    if(chance(gen) < flipMutationRate)
    {
        std::uniform_int_distribution<long int> flipDist(0, genome->synapses.size() - 1);
        int flipIdx = flipDist(gen);
        genome->synapses[flipIdx].enabled = !genome->synapses[flipIdx].enabled;
    }
}

void NEAT::reproduce()
{
    for(int i = 0; i < speciatedGenomes.size(); i++)
    {
        std::sort(speciatedGenomes[i].begin(), speciatedGenomes[i].end(),
                    [this](long int a, long int b) {return genomes[a].fitness > genomes[b].fitness;});

        int toAllocate = newGenomes[i].size();
        if(toAllocate == 0) continue;

        int current = 0;
        if(speciatedGenomes[i].size() > 5)
        {
            newGenomes[i][current] = genomes[speciatedGenomes[i][0]];
            current++;
        }

        long int topR = std::max((long int)1, (long int)(speciatedGenomes[i].size() * r));
        std::uniform_int_distribution<long int> parentDist(0, topR - 1);
        while(current < toAllocate)
        {
            genome* parent1 = &genomes[speciatedGenomes[i][parentDist(gen)]];
            genome* parent2 = &genomes[speciatedGenomes[i][parentDist(gen)]];
            newGenomes[i][current] = crossover(parent1, parent2);
            mutate(&newGenomes[i][current]);
            current++;
        }
    }

    genomes.clear();
    for(int i = 0; i < newGenomes.size(); i++)
    {
        for(int j = 0; j < newGenomes[i].size(); j++)
        {
            genomes.push_back(newGenomes[i][j]);
        }
    }
    populationSize = genomes.size();

    newGenomes.clear();
}

void NEAT::compileNetwork(genome& g)
{
    g.nnExecutionOrder.clear();
    std::unordered_map<long int, long int> inDegrees;
    std::unordered_map<long int, std::vector<std::pair<long int, float>>> adj;

    for(const synapseGene& synapse : g.synapses)
    {
        if(synapse.enabled)
        {
            adj[synapse.inputID].push_back({synapse.outputID, synapse.weight});
            inDegrees[synapse.outputID]++;
        }
    }

    std::queue<long int> ids;
    for(long int i = 0; i < g.inputs; i++)
    {
        if(inDegrees[i] == 0) ids.push(i);
    }
    for(long int id : g.neuronList)
    {
        if(id >= g.inputs && inDegrees[id] == 0) ids.push(id);
    }

    std::unordered_map<long int, long int> idToIdx;
    while(!ids.empty())
    {
        long int current = ids.front();
        ids.pop();
        idToIdx[current] = g.nnExecutionOrder.size();
        phenotypeNeuron newNeuron = {g.neurons[current].bias, g.neurons[current].type == OUTPUT, current};
        g.nnExecutionOrder.push_back(newNeuron);

        for(auto& neuron : adj[current])
        {
            inDegrees[neuron.first]--;
            if(inDegrees[neuron.first] == 0) ids.push(neuron.first);
        }
    }

    for(auto& neuron : g.nnExecutionOrder)
    {
        for(auto& outNeuron : adj[neuron.id]) neuron.outgoing.push_back({idToIdx[outNeuron.first], outNeuron.second});
    }
}

std::vector<float> NEAT::feedForward(genome& g, const std::vector<float>& inputs, float (*activationFunction)(float))
{    
    std::vector<float> nodeVals(g.nnExecutionOrder.size(), 0.0f);
    std::vector<float> outputVals(g.outputs, 0.0f);

    for(int i = 0; i < g.nnExecutionOrder.size(); i++)
    {
        const phenotypeNeuron& neuron = g.nnExecutionOrder[i];
        if(i < g.inputs) nodeVals[i] = inputs[i];
        else 
        {
            nodeVals[i] += neuron.bias;
            nodeVals[i] = activationFunction(nodeVals[i]);
        }

        if(neuron.isOutput)
        {
            outputVals[neuron.id - g.inputs] = nodeVals[i];
        }

        for(const phenotypeConnection& connection : neuron.outgoing)
            nodeVals[connection.targetNodeID] += nodeVals[i] * connection.weight;
    }

    return outputVals;
}

float NEAT::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}