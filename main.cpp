#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#define nfeatures 8
#define paramsfname "params.pt"

// Arch
/*
  8 inputs
  1 output (sigmoid, represents 2 classes)
  
  3 layers:
  8 -> 16
  16 -> 32
  32 -> 1
*/

struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(8, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 96));
    fc3 = register_module("fc3", torch::nn::Linear(96, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::leaky_relu(fc1 -> forward(x));
    x = torch::leaky_relu(fc2 -> forward(x));
    x = torch::sigmoid(fc3 -> forward(x));
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct CSVDataset : torch::data::datasets::Dataset<CSVDataset> {
  std::vector<torch::Tensor> data, targets;

  CSVDataset(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line, cell;

    std::getline(file, line);
    std::cout << line << std::endl;

    while (std::getline(file, line)) {
      std::stringstream line_stream(line);
      std::vector<float> values;

      while (std::getline(line_stream, cell, ',')) {
        values.push_back(std::stof(cell));
      }

      auto data_tensor = torch::tensor(std::vector<float>(values.begin(), values.end() - 1));
      auto target_tensor = torch::tensor(values.back(), torch::kFloat32);

      data.push_back(data_tensor);
      targets.push_back(target_tensor);
    }
  }

  torch::data::Example<> get(size_t index) override {
    return {data[index], targets[index]};
  }

  torch::optional<size_t> size() const override {
    return data.size();
  }
};


int main() {
  std::string choice;
  std::string flabels[nfeatures] = {
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve"
  };
  std::vector<float> sfeatures(nfeatures, 0.0f);
  std::string tmp;

  auto net = std::make_shared<Net>();

  do {
    std::cout << "Entrenamiento o inferencia? (e/i): ";
    std::cin >> choice;
    std::cout << "\n";
  } while (choice != "e" && choice != "i");

  if (choice == "e") {
    auto dataset = CSVDataset("/home/juanpa/Projects/cpp-projects/PredictingPulsarStars/pulsar_stars_dataset.csv").map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(std::move(dataset), /*batch_size=*/8192);
    

    torch::optim::Adam optimizer(net -> parameters(), /*lr=*/0.003);

    for (size_t epoch = 1; epoch <= 100; ++epoch) {
      size_t batch_index = 0;

      for (auto& batch : *data_loader) {
        optimizer.zero_grad();

        torch::Tensor prediction = net -> forward(batch.data); // forward pass
        auto target = batch.target.view({-1, 1});
        torch::Tensor loss = torch::nn::functional::binary_cross_entropy(prediction, target); // calc loss
        
        loss.backward(); // compute gradients
        optimizer.step(); // update model parameters

        batch_index++;

        if (1) { // ++batch_index % 100 == 0
          std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
        }
      }
    }

    torch::save(net, paramsfname);
  }
  if (choice == "i") {
    torch::load(net, paramsfname);
    std::cout << "Ingrese los datos requeridos\n";

    for (int i=0; i<nfeatures; i++) {
      std::cout << flabels[i] << ": ";
      std::cin >> tmp;
      sfeatures[i] = std::stof(tmp);
      std::cout << "Dato recibido: " << sfeatures[i] << "\n\n";
    }

    torch::Tensor tfeatures = torch::from_blob(sfeatures.data(), {1, static_cast<long>(sfeatures.size())});
    torch::Tensor prediction = net -> forward(tfeatures);
    std::cout << "Prediction: " << prediction << std::endl; 
  }
}
