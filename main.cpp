#include <torch/torch.h>
#include <iostream>

// Arch
/*
  8 inputs
  2 outputs (softmax, represents 2 classes)
  
  3 layers:
  8 -> 16
  16 -> 32
  32 -> 2
*/

struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(8, 16));
    fc2 = register_module("fc2", torch::nn::Linear(16, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 2));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1 -> forward(x));
    x = torch::relu(fc2 -> forward(x));
    x = torch::softmax(fc3 -> forward(x), /*dim=*/1);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


int main() {
  auto net = std::make_shared<Net>();
  std::cout << "Todo fino!" << std::endl;
}
