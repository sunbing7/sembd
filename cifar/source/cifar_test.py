




def test_attack_sr():
    model = load_model('/Users/bing.sun/workspace/Semantic/PyWorkplace/backdoor/cifar/models/cifar_semantic_greencar.h5')
    train_X, train_Y, test_X, test_Y = load_dataset()  # Load training and testing data
    base_gen = DataGenerator(None)
    test_adv_gen = base_gen.generate_data(test_X, test_Y, 3)  # Data generator for backdoor testing

    _, attack_acc = model.evaluate_generator(test_adv_gen, steps=5, verbose=1)
    print("Backdoor Success Rate {:.4f}".format(attack_acc))