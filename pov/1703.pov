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
    sphere { m*<0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 1 }        
    sphere {  m*<1.0990309764399133,9.86062196756703e-19,3.8053426322346677>, 1 }
    sphere {  m*<5.6256892977733655,5.364768824986535e-18,-1.1323054883989194>, 1 }
    sphere {  m*<-3.943893929673741,8.164965809277259,-2.2693997167119893>, 1}
    sphere { m*<-3.943893929673741,-8.164965809277259,-2.269399716711993>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0990309764399133,9.86062196756703e-19,3.8053426322346677>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5 }
    cylinder { m*<5.6256892977733655,5.364768824986535e-18,-1.1323054883989194>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5}
    cylinder { m*<-3.943893929673741,8.164965809277259,-2.2693997167119893>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5 }
    cylinder {  m*<-3.943893929673741,-8.164965809277259,-2.269399716711993>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5}

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
    sphere { m*<0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 1 }        
    sphere {  m*<1.0990309764399133,9.86062196756703e-19,3.8053426322346677>, 1 }
    sphere {  m*<5.6256892977733655,5.364768824986535e-18,-1.1323054883989194>, 1 }
    sphere {  m*<-3.943893929673741,8.164965809277259,-2.2693997167119893>, 1}
    sphere { m*<-3.943893929673741,-8.164965809277259,-2.269399716711993>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0990309764399133,9.86062196756703e-19,3.8053426322346677>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5 }
    cylinder { m*<5.6256892977733655,5.364768824986535e-18,-1.1323054883989194>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5}
    cylinder { m*<-3.943893929673741,8.164965809277259,-2.2693997167119893>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5 }
    cylinder {  m*<-3.943893929673741,-8.164965809277259,-2.269399716711993>, <0.9401153127265357,-1.9843804022266337e-18,0.8095485264720534>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    