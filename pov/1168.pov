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
    sphere { m*<0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 1 }        
    sphere {  m*<0.2675029459377681,-1.2264477029113265e-18,4.11097235015932>, 1 }
    sphere {  m*<8.516933904868148,5.499450044079284e-18,-1.9157057200570649>, 1 }
    sphere {  m*<-4.513282176624976,8.164965809277259,-2.172137411260879>, 1}
    sphere { m*<-4.513282176624976,-8.164965809277259,-2.1721374112608824>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2675029459377681,-1.2264477029113265e-18,4.11097235015932>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5 }
    cylinder { m*<8.516933904868148,5.499450044079284e-18,-1.9157057200570649>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5}
    cylinder { m*<-4.513282176624976,8.164965809277259,-2.172137411260879>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5 }
    cylinder {  m*<-4.513282176624976,-8.164965809277259,-2.1721374112608824>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5}

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
    sphere { m*<0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 1 }        
    sphere {  m*<0.2675029459377681,-1.2264477029113265e-18,4.11097235015932>, 1 }
    sphere {  m*<8.516933904868148,5.499450044079284e-18,-1.9157057200570649>, 1 }
    sphere {  m*<-4.513282176624976,8.164965809277259,-2.172137411260879>, 1}
    sphere { m*<-4.513282176624976,-8.164965809277259,-2.1721374112608824>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2675029459377681,-1.2264477029113265e-18,4.11097235015932>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5 }
    cylinder { m*<8.516933904868148,5.499450044079284e-18,-1.9157057200570649>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5}
    cylinder { m*<-4.513282176624976,8.164965809277259,-2.172137411260879>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5 }
    cylinder {  m*<-4.513282176624976,-8.164965809277259,-2.1721374112608824>, <0.23576508052761105,-2.2366200452431323e-18,1.1111392422740605>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    