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
    sphere { m*<-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 1 }        
    sphere {  m*<-1.6331118432856048e-19,-1.5302228469966917e-19,9.994296211354573>, 1 }
    sphere {  m*<9.428090415820634,-1.541937382545121e-19,-3.332037121978761>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.332037121978761>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.332037121978761>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.6331118432856048e-19,-1.5302228469966917e-19,9.994296211354573>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5 }
    cylinder { m*<9.428090415820634,-1.541937382545121e-19,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5}

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
    sphere { m*<-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 1 }        
    sphere {  m*<-1.6331118432856048e-19,-1.5302228469966917e-19,9.994296211354573>, 1 }
    sphere {  m*<9.428090415820634,-1.541937382545121e-19,-3.332037121978761>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.332037121978761>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.332037121978761>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.6331118432856048e-19,-1.5302228469966917e-19,9.994296211354573>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5 }
    cylinder { m*<9.428090415820634,-1.541937382545121e-19,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.332037121978761>, <-1.0976549906589814e-20,-2.553888456355728e-19,0.0012962113545725166>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    