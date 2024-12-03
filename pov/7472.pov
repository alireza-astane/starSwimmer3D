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
    sphere { m*<-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 1 }        
    sphere {  m*<0.8330467574798218,0.19294216164226197,9.328300875553644>, 1 }
    sphere {  m*<8.200833955802624,-0.09215008914999956,-5.242376553520286>, 1 }
    sphere {  m*<-6.695129237886368,6.430931284470638,-3.751569650338679>, 1}
    sphere { m*<-3.1718091836754327,-6.428121313828509,-1.7183893478189485>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8330467574798218,0.19294216164226197,9.328300875553644>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5 }
    cylinder { m*<8.200833955802624,-0.09215008914999956,-5.242376553520286>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5}
    cylinder { m*<-6.695129237886368,6.430931284470638,-3.751569650338679>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5 }
    cylinder {  m*<-3.1718091836754327,-6.428121313828509,-1.7183893478189485>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5}

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
    sphere { m*<-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 1 }        
    sphere {  m*<0.8330467574798218,0.19294216164226197,9.328300875553644>, 1 }
    sphere {  m*<8.200833955802624,-0.09215008914999956,-5.242376553520286>, 1 }
    sphere {  m*<-6.695129237886368,6.430931284470638,-3.751569650338679>, 1}
    sphere { m*<-3.1718091836754327,-6.428121313828509,-1.7183893478189485>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8330467574798218,0.19294216164226197,9.328300875553644>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5 }
    cylinder { m*<8.200833955802624,-0.09215008914999956,-5.242376553520286>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5}
    cylinder { m*<-6.695129237886368,6.430931284470638,-3.751569650338679>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5 }
    cylinder {  m*<-3.1718091836754327,-6.428121313828509,-1.7183893478189485>, <-0.5861207367203402,-0.7969967522376554,-0.5209892214815043>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    