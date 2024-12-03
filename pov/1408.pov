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
    sphere { m*<0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 1 }        
    sphere {  m*<0.6449213328595075,-2.127532096590517e-18,3.9815831488250337>, 1 }
    sphere {  m*<7.221247952981312,1.9142558771724895e-18,-1.5829311863468762>, 1 }
    sphere {  m*<-4.243723254524822,8.164965809277259,-2.2179818750780944>, 1}
    sphere { m*<-4.243723254524822,-8.164965809277259,-2.217981875078098>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6449213328595075,-2.127532096590517e-18,3.9815831488250337>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5 }
    cylinder { m*<7.221247952981312,1.9142558771724895e-18,-1.5829311863468762>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5}
    cylinder { m*<-4.243723254524822,8.164965809277259,-2.2179818750780944>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5 }
    cylinder {  m*<-4.243723254524822,-8.164965809277259,-2.217981875078098>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5}

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
    sphere { m*<0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 1 }        
    sphere {  m*<0.6449213328595075,-2.127532096590517e-18,3.9815831488250337>, 1 }
    sphere {  m*<7.221247952981312,1.9142558771724895e-18,-1.5829311863468762>, 1 }
    sphere {  m*<-4.243723254524822,8.164965809277259,-2.2179818750780944>, 1}
    sphere { m*<-4.243723254524822,-8.164965809277259,-2.217981875078098>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6449213328595075,-2.127532096590517e-18,3.9815831488250337>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5 }
    cylinder { m*<7.221247952981312,1.9142558771724895e-18,-1.5829311863468762>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5}
    cylinder { m*<-4.243723254524822,8.164965809277259,-2.2179818750780944>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5 }
    cylinder {  m*<-4.243723254524822,-8.164965809277259,-2.217981875078098>, <0.5613519566435926,-5.8483018654093e-18,0.9827444821111289>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    