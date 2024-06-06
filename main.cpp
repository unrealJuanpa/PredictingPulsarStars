#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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
  auto dataset = CSVDataset("/home/juanpa/Projects/cpp-projects/PredictingPulsarStars/pulsar_stars_dataset.csv").map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset), /*batch_size=*/8192);
  auto net = std::make_shared<Net>();

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
}
