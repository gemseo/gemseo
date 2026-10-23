function change_group_name(e,index) {
  let txt = e.innerHTML;
  if (mat.collapsed_groups.includes(index)) {
      new_index = mat.collapsed_groups.indexOf(index);
      d3.select("#rowname"+new_index).text(txt).insert("title").text(txt);
      d3.select("#colname"+new_index).text(txt).insert("title").text(txt);
      mat.nodes[new_index].name = txt;
      mat.nodes[new_index].description = mat.nodes[new_index].description.replace(/>.*<\/h2>/, ">"+txt+"</h2>");
  }
  mat.groups[index] = txt;
  mat.json.nodes[index].name = txt;
  mat.json.nodes[index].description = mat.json.nodes[index].description.replace(/>.*<\/h2>/, ">"+txt+"</h2>");
  mat.json.groups[index] = txt;
}

document.addEventListener('keydown', function (event) {
  var escape = event.which == 27,
      enter = event.which == 13,
      element = event.target,
      input = element.nodeName == 'SPAN',
      data = {};

  if (input) {
    if (escape) {
      document.execCommand('undo');
      element.blur();
    } else if (enter) {
      element.blur();
      event.preventDefault();
    }
  }
}, true);
