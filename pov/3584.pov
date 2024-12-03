#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 1 }        
    sphere {  m*<0.33251556497207674,0.510086282182,2.9126813334413186>, 1 }
    sphere {  m*<2.826488854236643,0.483410179388049,-1.3040829631304165>, 1 }
    sphere {  m*<-1.529834899662506,2.709850148420276,-1.0488192030952024>, 1}
    sphere { m*<-2.5730045930826497,-4.656017888699374,-1.6188330813113248>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33251556497207674,0.510086282182,2.9126813334413186>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5 }
    cylinder { m*<2.826488854236643,0.483410179388049,-1.3040829631304165>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5}
    cylinder { m*<-1.529834899662506,2.709850148420276,-1.0488192030952024>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5 }
    cylinder {  m*<-2.5730045930826497,-4.656017888699374,-1.6188330813113248>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 1 }        
    sphere {  m*<0.33251556497207674,0.510086282182,2.9126813334413186>, 1 }
    sphere {  m*<2.826488854236643,0.483410179388049,-1.3040829631304165>, 1 }
    sphere {  m*<-1.529834899662506,2.709850148420276,-1.0488192030952024>, 1}
    sphere { m*<-2.5730045930826497,-4.656017888699374,-1.6188330813113248>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33251556497207674,0.510086282182,2.9126813334413186>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5 }
    cylinder { m*<2.826488854236643,0.483410179388049,-1.3040829631304165>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5}
    cylinder { m*<-1.529834899662506,2.709850148420276,-1.0488192030952024>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5 }
    cylinder {  m*<-2.5730045930826497,-4.656017888699374,-1.6188330813113248>, <0.09178046023038522,0.3813762040016746,-0.07487343767923083>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    