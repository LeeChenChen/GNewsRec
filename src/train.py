import tensorflow as tf
import os
import numpy as np
from model import Model
from data_loader import train_random_neighbor, test_random_neighbor


def train(args, data, show_loss):
    train_data, eval_data, test_data = data[0], data[1], data[2]
    # user_news, news_user, news_title = data[3], data[4], data[5]
    # news_entity, entity_news, clicks = data[6], data[7], data[8]
    # entity_entity = data[9]
    train_user_news, train_news_user, test_user_news, test_news_user, topic_news = data[3], data[4], data[5], data[6], data[7]
    news_topic, news_title, news_entity, news_group = data[8], data[9], data[10], data[11]
    #train_clicks, test_clicks = data[12], data[13]

    # model = Model(args, user_news, news_user, news_title, news_entity, entity_news, entity_entity)
    model = Model(args, news_title, news_entity, news_group, news_topic, len(train_user_news), len(news_title))

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)#,log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        file = open("local-" + str(args.n_topics) + ".txt", "a")
        global_step = 0
        for step in range(args.n_epochs):
            np.random.shuffle(train_data)
            start = 0
            max_f1 = 0
            user_news, news_user, _topic_news = train_random_neighbor(args, train_user_news, train_news_user,
                                                                      topic_news, len(news_title))
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                global_step += 1
                clicks_news = [x + [0]*(args.session_len-len(x)) if len(x) < args.session_len else x[-args.session_len:] for x in train_data[start:start + args.batch_size, 2]]


                _, loss,n,u = model.train(sess, get_feed_dict(model, train_data, start,
                                                                                start + args.batch_size, clicks_news,
                                                                                0.5,user_news,news_user,_topic_news))
                start += args.batch_size

                if start % 12800 == 0:
                    train_auc, train_f1,p,r = ctr_eval(sess, model, train_data[:2048], args.batch_size, args,
                                                   train_user_news, train_news_user, topic_news, len(news_title))
                    eval_auc, eval_f1,p1,r1 = ctr_eval(sess, model, eval_data, args.batch_size, args, test_user_news, test_news_user,topic_news, len(news_title))
                    print(train_auc, train_f1, eval_auc, eval_f1)
                # if show_loss:
                    print(start, loss)

                    file.write(str(eval_auc)+" " +str(eval_f1) + "\n")
                    file.write(str(start) + " " + str(loss) + "\n")
                    if eval_f1 >= max_f1:
                        saver.save(sess, os.path.join(args.save_path, 'model.ckpt'), global_step=global_step)
                        max_f1 = eval_f1

            # CTR evaluation
            train_auc, train_f1,p,r = ctr_eval(sess, model, train_data[:2048], args.batch_size, args,train_user_news, train_news_user, topic_news, len(news_title))

            eval_auc, eval_f1,p,r = ctr_eval(sess, model, eval_data, args.batch_size, args,test_user_news, test_news_user,topic_news, len(news_title))
            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1))
            test_auc, test_f1,p,r = ctr_eval(sess, model, test_data, args.batch_size, args,test_user_news, test_news_user,topic_news, len(news_title))
            print('test auc: %.4f  f1: %.4f'
                  % (test_auc, test_f1))
            file.write("\n-----------\n"+str(step) + "\n" + str(train_auc)+ " " + str(train_f1)+ " " + str(eval_auc)+ " "+str(eval_f1) + " " + str(test_auc)+ " " + str(test_f1) + "\n")
        file.close()

def test(args, data):
    test_data = data[2]
    train_user_news, train_news_user, test_user_news, test_news_user, topic_news = data[3], data[4], data[5], data[6], \
                                                                                   data[7]
    news_topic, news_title, news_entity, news_group = data[8], data[9], data[10], data[11]
    test_data_old,test_data_new = data[12],data[13]
    model = Model(args, news_title, news_entity, news_group, news_topic, len(train_user_news), len(news_title))

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        moudke_file = tf.train.latest_checkpoint(args.save_path)
        saver.restore(sess, moudke_file)
        test_auc, test_f1,test_p,test_r = ctr_eval(sess, model, test_data, args.batch_size, args, test_user_news, test_news_user,
                                     topic_news, len(news_title))

        print('test auc: %.4f  f1: %.4f  p: %.4f  r: %.4f' % (test_auc, test_f1,test_p,test_r))



def get_feed_dict(model, data, start, end, clicks_news,droupout,user_news,news_user,topic_news):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.news_indices: data[start:end, 1],
                 model.dropout_rate: droupout,
                 model.clicks: clicks_news,
                 model.labels: data[start:end, 3],
                 model.user_news: user_news,
                 model.news_user: news_user,
                 model.topic_news: topic_news}
    return feed_dict


def ctr_eval(sess, model, data, batch_size, args,train_user_news, train_news_user,topic_news, len_n):
    start = 0
    auc_list = []
    f1_list = []
    p_list = []
    r_list = []
    user_news, news_user, _topic_news = test_random_neighbor(args, train_user_news, train_news_user,
                                                             topic_news, len_n)
    while start + batch_size <= data.shape[0]:
        clicks_news = [
            x + [0] * (args.session_len - len(x)) if len(x) < args.session_len else x[-args.session_len:]
            for x in data[start:start + args.batch_size, 2]]



        auc, f1, p,r = model.eval(sess, get_feed_dict(model, data, start, start + batch_size, clicks_news, 0,user_news,news_user,_topic_news))
        auc_list.append(auc)
        f1_list.append(f1)
        p_list.append(p)
        r_list.append(r)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)),float(np.mean(p_list)), float(np.mean(r_list))
