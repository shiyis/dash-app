window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, layer) {
                layer.on('click', function(e) {
                    this.bringToBack(); // Example JavaScript action
                    console.log("Custom JavaScript injection successful!");
                });
            }

            ,
        function(feature, layer) {
            layer.on('popupopen', function(e) {
                e.popup._container.style.zIndex = 1000;  // Bring to front
            });
        }
    }
});
