import h5py as h5

def dump_h5(obj, max_depth = 1000, depth=0, override_name=None):
    """Dump names of groups and datasets in an h5 file."""
    if obj.name == '/': depth = -1
    key = obj.name.split('/')[-1]
    printed_name = str(override_name if override_name is not None else key)
    if isinstance(obj, h5.Dataset):
        attrs = ["'%s'" % attr for attr in obj.attrs.keys()]        
        print('%s- %s (%s)' % (' '*depth, printed_name, ', '.join(attrs)))
    elif isinstance(obj, h5.Group):
        if obj.name != '/':
            print('%s> %s' % (' '*depth, printed_name))
        len_obj_keys = len(obj.keys())
        use_compact_display = len_obj_keys > 20
        for i, key in enumerate(obj.keys()):
            if depth < max_depth :
                if( not use_compact_display or ( i < 3 or i >= len_obj_keys - 3) ):
                    dump_h5(obj[key], max_depth, depth+1)
                elif i == 3:
                    print('%s - ...' % (' '*depth))
    else:
        error('H5 file should only contains Group and Datasets...')
