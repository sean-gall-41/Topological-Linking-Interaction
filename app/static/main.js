var scene, renderer, camera;
var controls;

function init() {
    // create the scene. This will hold our polymers
    scene = new THREE.Scene();
    var width = window.innerWidth;
    var height = window.innerHeight;
    // create the camera, which will be used to view our polymers
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

    // create RENDERER!!!
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(new THREE.Color(0x1, 1.0));
    renderer.setSize(width, height);
    renderer.shadowMapEnabled = true; //not sure what this does

    document.getElementById("WebGL-output").appendChild(renderer.domElement);

    // create axes
    var axes = new THREE.AxisHelper(10);
    scene.add(axes);

    var group = new THREE.Object3D();
    var chain_geometry = new THREE.Geometry();

    // obtain the actual chain data from python
    fetch('/data_helper')
        .then(function (response) {
            return response.json();
        }).then(function (json) {
            // Report that we got the payload...
            console.log('GET response JSON:');
            console.log(json);
            // Now do something with it
            var obj = JSON.parse(json);
            var vertexArray = obj.vertices;

            vertexArray.forEach((vertex) => {
                chain_geometry.vertices.push(
                    new THREE.Vector3(vertex[0], vertex[1], vertex[2])
                );
            });

            // now create the mesh
            for (var j = 0; j < chain_geometry.vertices.length - 1; ++j) {
                var path = new THREE.SplineCurve3([chain_geometry.vertices[j], chain_geometry.vertices[j + 1]]);
                var tube = new THREE.TubeGeometry(path, 1, 0.04);
                var material = new THREE.MeshPhongMaterial({ color: 0xcccccc });
                var mesh = new THREE.Mesh(tube, material);
                group.add(mesh);
            }
            scene.add(group);
        });


    // position and point the camera to the center of the scene initially
    camera.position.x = 6;
    camera.position.y = 6;
    camera.position.z = 6;
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    controls = new THREE.OrbitControls(camera, renderer.domElement);

    // add spotlight for the shadows
    var spotLight = new THREE.SpotLight(0xffffff);
    spotLight.position.set(-40, 60, -10);
    scene.add(spotLight);
}

// this is the mesh which tells the renderer what to do with
// a given geometry
function createMesh(geometry) {
    var geometryMaterial = new THREE.MeshBasicMaterial({ color: 0xF6DBD5, wireframe: true });
    var mesh = new THREE.Mesh(geometry, geometryMaterial);

    return mesh;
}

function animate() {
    controls.update();
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

init();
animate();
