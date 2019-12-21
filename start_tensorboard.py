from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'logs/cow'])
url = tb.launch()