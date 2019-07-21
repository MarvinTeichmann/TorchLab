from . import fcn


def get_decoder(conf, *args, **kwargs):

    decoder_type = conf['decoder']['type']

    if decoder_type == "linear":
        from torchlab.decoder.linear import LinearDecoder
        return LinearDecoder(conf, *args, **kwargs)
    else:
        raise NotImplementedError
