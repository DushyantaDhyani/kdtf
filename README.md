# Knowledge Distillation - Tensorflow

This is an implementation for the basic idea behind Hinton's [Knowledge Distillation Paper][1]. We do not reproduce the exact results but rather show that the idea works.

While a few other implementations are available, the code flow is not very intuitive. Here we generate the soft targets from the teacher in an on-line manner while training the student network.

The big and small models (with some modification - We currently have a simple softmax regression as in [TF's tutorial](https://www.tensorflow.org/versions/r1.2/get_started/mnist/beginners)) have been taken from [here](https://github.com/aymericdamien/TensorFlow-Examples). 

While this may not (or may) be a good way to implement the distillation architecture, it leads to a good improvement in the (small) student model. In case you find any bug or have any suggestions feel free to create an issue or even send in a pull request.

### Requirements

Tensorflow 1.3 or above

### Running the code

Train the Teacher Model

     python main.py --model_type teacher --checkpoint_dir teachercpt --num_steps 5000 --temperature 5
     
Train the Student Model (in a standalone manner for comparison)

     python main.py --model_type student --checkpoint_dir studentcpt --num_steps 5000
     
Train the Student Model (Using Soft Targets from the teacher model)

     python main.py --model_type student --checkpoint_dir studentcpt --load_teacher_from_checkpoint true --load_teacher_checkpoint_dir teachercpt --num_steps 5000 --temperature 5
     
### Results

| Model        | Accuracy  - 2 | Accuracy - 5 |
| -------------|:-------------:| -------------|  
| Teacher Only | 97.9          | 98.12        |      
| Distillation | 89.14         |  90.77       |  
| Student Only | 88.84         | 88.84        | 

The small model when trained without the soft labels always use **temperature**=1.

### References

[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)





[1]: https://arxiv.org/abs/1503.02531