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
    sphere { m*<1.1326711355385854,0.25368966362404055,0.5355779482710977>, 1 }        
    sphere {  m*<1.3768251312274715,0.27310284418401437,3.525562387747402>, 1 }
    sphere {  m*<3.8700723202900082,0.27310284418401437,-0.6917198207432147>, 1 }
    sphere {  m*<-3.2442150841208757,7.276118073944056,-2.0523243112558776>, 1}
    sphere { m*<-3.757593505576763,-7.975467316319444,-2.3551861588978547>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3768251312274715,0.27310284418401437,3.525562387747402>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5 }
    cylinder { m*<3.8700723202900082,0.27310284418401437,-0.6917198207432147>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5}
    cylinder { m*<-3.2442150841208757,7.276118073944056,-2.0523243112558776>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5 }
    cylinder {  m*<-3.757593505576763,-7.975467316319444,-2.3551861588978547>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5}

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
    sphere { m*<1.1326711355385854,0.25368966362404055,0.5355779482710977>, 1 }        
    sphere {  m*<1.3768251312274715,0.27310284418401437,3.525562387747402>, 1 }
    sphere {  m*<3.8700723202900082,0.27310284418401437,-0.6917198207432147>, 1 }
    sphere {  m*<-3.2442150841208757,7.276118073944056,-2.0523243112558776>, 1}
    sphere { m*<-3.757593505576763,-7.975467316319444,-2.3551861588978547>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3768251312274715,0.27310284418401437,3.525562387747402>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5 }
    cylinder { m*<3.8700723202900082,0.27310284418401437,-0.6917198207432147>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5}
    cylinder { m*<-3.2442150841208757,7.276118073944056,-2.0523243112558776>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5 }
    cylinder {  m*<-3.757593505576763,-7.975467316319444,-2.3551861588978547>, <1.1326711355385854,0.25368966362404055,0.5355779482710977>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    