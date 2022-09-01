import orjson
import logging
import ptgctl
import ptgctl.holoframe
import ptgctl.util
import numpy as np
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

configs = {'rule_classifier_path': '/Users/rlopez/PTG/experiments/models/recipe_tagger',
           'bert_classifier_path': '/Users/rlopez/PTG/experiments/models/bert_classifier/'}


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username='reasoning', password='reasoning')

    @ptgctl.util.async2sync
    async def run(self):
        input_sid = 'clip:basic-zero-shot:instructions'

        step_id_repr = 'reasoning:step_id'
        step_status_repr = 'reasoning:step_status'
        step_description_repr = 'reasoning:step_description'
        error_status_repr = 'reasoning:error_status'
        error_description_repr = 'reasoning:error_description'

        output_sids = [step_id_repr, step_status_repr, step_description_repr, error_status_repr, error_description_repr]

        recipe_id: str = self.api.sessions.current_recipe()
        recipe = self.api.recipes.get(recipe_id)
        logger.info('Loaded recipe: %s' % str(recipe))
        state_manager = StateManager(recipe, configs)
        step_data = state_manager.start_steps()
        logger.info('First step: %s' % str(step_data))

        async with self.api.data_pull_connect(input_sid) as ws_pull, \
                   self.api.data_push_connect(output_sids) as ws_push:
            while True:
                print('Inside')
                data = ptgctl.holoframe.load_all(await ws_pull.recv_data())
                logger.info('Data  from Hololens: %s' % str(data))
                action_pred = data[input_sid]
                action_text, action_prob = zip(*action_pred.items())
                logger.info('Perception outputs: %s' % str(action_text))
                i_topk = np.argpartition(action_prob, 5)
                i_topk = sorted(i_topk, key=lambda i: action_prob[i], reverse=True)

                results = state_manager.check_status([action_text[i] for i in i_topk])

                await ws_push.send_data(
                    [
                        orjson.dumps(results['step_id']),
                        orjson.dumps(results['step_status']),
                        orjson.dumps(results['step_description']),
                        orjson.dumps(results['error_status']),
                        orjson.dumps(results['error_description'])
                    ]
                )


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)
