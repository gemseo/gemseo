function save_json() {
    var blob = new Blob([JSON.stringify(mat.json)], {type: "text/plain;charset=utf-8"});
    var filename = prompt("Please enter a file name to which the extension .json will be added:", "n2");
    if (filename != null) {
        saveAs(blob, filename + '.json');
    }
}
