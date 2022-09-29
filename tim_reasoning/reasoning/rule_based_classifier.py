import sys
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class RuleBasedClassifier:

    def __init__(self, model_tagger):
        self.recipe_tagger = model_tagger
        self.nlp = spacy.load('en_core_web_lg')

    def is_mistake(self, current_step, detected_action, threshold=0.8):
        tokens, tags = self.recipe_tagger.predict_entities(current_step)
        step_action_pairs = self.recipe_tagger.extract_action_relations(tokens, tags)

        tokens, tags = self.recipe_tagger.predict_entities(detected_action)
        detected_action_pairs = self.recipe_tagger.extract_action_relations(tokens, tags)

        logger.info('Pair of (action, object) for the recipe step: %s', str(step_action_pairs))
        logger.info('Pair of (action, object) for the detected actions: %s', str(detected_action_pairs))

        for step_action, step_object in step_action_pairs:
            step_text = step_action + ' ' + step_object if step_object is not None else step_action
            text_query = self.nlp(step_text)
            for detected_action, detected_object in detected_action_pairs:
                detected_text = detected_action + ' ' + detected_object if detected_object is not None else detected_action
                similarity = text_query.similarity(self.nlp(detected_text))
                logger.info('Similarity between %s and %s: (%.2f)' % (step_text, detected_text, similarity))
                if similarity >= threshold:
                    logger.info('It is not a mistake')
                    return False

        logger.info('It is a mistake')

        return True
