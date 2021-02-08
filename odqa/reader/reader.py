from abc import ABC


class Reader(ABC):
    """
    Abstract class for all the readers
    """

    @staticmethod
    def span_candidates(answers, context, tokenizer):
        """Check context for span of answer
        Args:
            answers: [[answer]]
            context: ["input_ids"]
        Returns:
            span: <int32> [(start,end)]
        """

        tokenized_answers = tokenizer(answers)

        # Remove [CLS] and [SEP] tokens from answers
        answers = []
        for answer in tokenized_answers["input_ids"]:
            answers.append(answer[1:-1])

        span = []
        found = False
        for answer in answers:
            length=len(answer)
            for ind in (i for i,e in enumerate(context) if e==answer[0]):
                if context[ind:ind+length]==answer:
                    span.append((ind,ind+length-1))
                    found = True
                    continue
        # CLS index if not found
        if not found:
            span.append((0,0))

        return span