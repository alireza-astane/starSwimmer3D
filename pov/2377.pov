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
    sphere { m*<0.9858227308170088,0.4949768560646637,0.44875129743894526>, 1 }        
    sphere {  m*<1.2296653374933038,0.5353486936386681,3.4385507563976185>, 1 }
    sphere {  m*<3.722912526555842,0.5353486936386679,-0.7787314520930011>, 1 }
    sphere {  m*<-2.7885371388364972,6.398812169667681,-1.782891422336944>, 1}
    sphere { m*<-3.8183612608886857,-7.8023400748206315,-2.391119246863039>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2296653374933038,0.5353486936386681,3.4385507563976185>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5 }
    cylinder { m*<3.722912526555842,0.5353486936386679,-0.7787314520930011>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5}
    cylinder { m*<-2.7885371388364972,6.398812169667681,-1.782891422336944>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5 }
    cylinder {  m*<-3.8183612608886857,-7.8023400748206315,-2.391119246863039>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5}

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
    sphere { m*<0.9858227308170088,0.4949768560646637,0.44875129743894526>, 1 }        
    sphere {  m*<1.2296653374933038,0.5353486936386681,3.4385507563976185>, 1 }
    sphere {  m*<3.722912526555842,0.5353486936386679,-0.7787314520930011>, 1 }
    sphere {  m*<-2.7885371388364972,6.398812169667681,-1.782891422336944>, 1}
    sphere { m*<-3.8183612608886857,-7.8023400748206315,-2.391119246863039>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2296653374933038,0.5353486936386681,3.4385507563976185>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5 }
    cylinder { m*<3.722912526555842,0.5353486936386679,-0.7787314520930011>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5}
    cylinder { m*<-2.7885371388364972,6.398812169667681,-1.782891422336944>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5 }
    cylinder {  m*<-3.8183612608886857,-7.8023400748206315,-2.391119246863039>, <0.9858227308170088,0.4949768560646637,0.44875129743894526>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    