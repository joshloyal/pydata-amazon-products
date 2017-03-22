var data = source.get('data');
var original_data = cb_obj.get('data');
var inds = cb_obj.get('selected')['1d'].indices;

for (var key in original_data) {
    data[key] = [];
    for (var i = 0; i < original_data['%s'].length; ++i) {
        if (inds.indexOf(i) > -1) {
            data[key].push(original_data[key][i]);
        }
    }
}

target_obj.trigger('change')
source.trigger('change')
