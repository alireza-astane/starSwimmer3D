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
    sphere { m*<-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 1 }        
    sphere {  m*<0.6851059677643386,-0.12924399176265955,9.259791337792288>, 1 }
    sphere {  m*<8.052893166087136,-0.4143362425549222,-5.310886091281645>, 1 }
    sphere {  m*<-6.8430700276018515,6.108745131065731,-3.820079188100041>, 1}
    sphere { m*<-2.427808324045131,-4.80783271353619,-1.3738518130028077>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6851059677643386,-0.12924399176265955,9.259791337792288>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5 }
    cylinder { m*<8.052893166087136,-0.4143362425549222,-5.310886091281645>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5}
    cylinder { m*<-6.8430700276018515,6.108745131065731,-3.820079188100041>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5 }
    cylinder {  m*<-2.427808324045131,-4.80783271353619,-1.3738518130028077>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5}

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
    sphere { m*<-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 1 }        
    sphere {  m*<0.6851059677643386,-0.12924399176265955,9.259791337792288>, 1 }
    sphere {  m*<8.052893166087136,-0.4143362425549222,-5.310886091281645>, 1 }
    sphere {  m*<-6.8430700276018515,6.108745131065731,-3.820079188100041>, 1}
    sphere { m*<-2.427808324045131,-4.80783271353619,-1.3738518130028077>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6851059677643386,-0.12924399176265955,9.259791337792288>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5 }
    cylinder { m*<8.052893166087136,-0.4143362425549222,-5.310886091281645>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5}
    cylinder { m*<-6.8430700276018515,6.108745131065731,-3.820079188100041>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5 }
    cylinder {  m*<-2.427808324045131,-4.80783271353619,-1.3738518130028077>, <-0.7340615264358236,-1.1191829056425773,-0.5894987592428639>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    