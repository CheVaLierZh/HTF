

class BaseRestorer(object):
    def restore(self, data, mask):
        """

        :param data: (n_variates, n_timepoints) data with missings to be restored, content is changed after this method,
            copy if needed
        :param mask: (n_variates, n_timepoints) mask of data indicating which element is missing, marked as 0, otherwise 1
        :return: (n_variates, n_timepoints) restored data with missing position is replaced with imputed data
        """
        pass
