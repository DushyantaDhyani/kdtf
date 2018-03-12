import os
import argparse
import model
import data
import tensorflow as tf
import numpy as np
import random

seed = int(os.getenv("SEED", 12))
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="false")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)

    # Model Parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropoutprob', type=float, default=0.75)

    return parser


def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)

    check_and_makedir(args.log_dir)
    check_and_makedir(args.checkpoint_dir)


def main():
    parser = get_parser()
    args = parser.parse_args()
    setup(args)
    dataset = data.Dataset(args)

    tf.reset_default_graph()
    if args.model_type == "student":
        teacher_model = None
        if args.load_teacher_from_checkpoint:
            teacher_model = model.BigModel(args, "teacher")
            teacher_model.start_session()
            teacher_model.load_model_from_file(args.load_teacher_checkpoint_dir)
            print("Verify Teacher State before Training Student")
            teacher_model.run_inference(dataset)
        student_model = model.SmallModel(args, "student")
        student_model.start_session()
        student_model.train(dataset, teacher_model)

        # Testing student model on the best model based on validation set
        student_model.load_model_from_file(args.checkpoint_dir)
        student_model.run_inference(dataset)

        if args.load_teacher_from_checkpoint:
            print("Verify Teacher State After Training student Model")
            teacher_model.run_inference(dataset)
            teacher_model.close_session()
        student_model.close_session()
    else:
        teacher_model = model.BigModel(args, "teacher")
        teacher_model.start_session()
        teacher_model.train(dataset)

        # Testing teacher model on the best model based on validation set
        teacher_model.load_model_from_file(args.checkpoint_dir)
        teacher_model.run_inference(dataset)
        teacher_model.close_session()


if __name__ == '__main__':
    main()
    # INVOCATION

    # Teacher
    # python main.py --model_type teacher --checkpoint_dir teachercpt --num_steps 50

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --num_steps 50 --gpu 0

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --load_teacher_from_checkpoint true --load_teacher_checkpoint_dir teachercpt --num_steps 50
