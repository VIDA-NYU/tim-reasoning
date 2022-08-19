from tim_reasoning.reasoning.recipe_tagger import RecipeTagger
import sys
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_lg')


class RuleBasedClassifier:

    def __init__(self, model_tagger_path):
        self.recipe_tagger = RecipeTagger(model_tagger_path)

    def is_mistake(self, current_step, detected_actions, threshold=0.9):
        tokens, tags = self.recipe_tagger.predict_entities(current_step)
        step_action_pairs = self.recipe_tagger.extract_action_relations(tokens, tags)

        detected_action_pairs = []
        for detected_action in detected_actions:
            tokens, tags = self.recipe_tagger.predict_entities(detected_action)
            detected_action_pairs += self.recipe_tagger.extract_action_relations(tokens, tags)

        logger.info('Pair of (action, object) for the recipe step: %s', str(step_action_pairs))
        logger.info('Pair of (action, object) for the detected actions: %s', str(detected_action_pairs))

        for step_action, step_object in step_action_pairs:
            step_text = step_action + ' ' + step_object if step_object is not None else step_action
            text_query = nlp(step_text)
            for detected_action, detected_object in detected_action_pairs:
                detected_text = detected_action + ' ' + detected_object if detected_object is not None else detected_action
                similarity = text_query.similarity(nlp(detected_text))
                if similarity >= threshold:
                    logger.info('Found similar pairs: %s and %s (%.2f)' % (step_text, detected_text, similarity))
                    logger.info('It is not a mistake!')
                    return False

        logger.info('It is a mistake!')

        return True
