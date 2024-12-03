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
    sphere { m*<0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 1 }        
    sphere {  m*<0.5101610603496166,-2.061481075862789e-18,4.029293598560879>, 1 }
    sphere {  m*<7.685760921185633,2.7193182911058937e-18,-1.7049701696340829>, 1 }
    sphere {  m*<-4.337997519064569,8.164965809277259,-2.2019807507836546>, 1}
    sphere { m*<-4.337997519064569,-8.164965809277259,-2.2019807507836573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5101610603496166,-2.061481075862789e-18,4.029293598560879>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5 }
    cylinder { m*<7.685760921185633,2.7193182911058937e-18,-1.7049701696340829>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5}
    cylinder { m*<-4.337997519064569,8.164965809277259,-2.2019807507836546>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5 }
    cylinder {  m*<-4.337997519064569,-8.164965809277259,-2.2019807507836573>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5}

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
    sphere { m*<0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 1 }        
    sphere {  m*<0.5101610603496166,-2.061481075862789e-18,4.029293598560879>, 1 }
    sphere {  m*<7.685760921185633,2.7193182911058937e-18,-1.7049701696340829>, 1 }
    sphere {  m*<-4.337997519064569,8.164965809277259,-2.2019807507836546>, 1}
    sphere { m*<-4.337997519064569,-8.164965809277259,-2.2019807507836573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5101610603496166,-2.061481075862789e-18,4.029293598560879>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5 }
    cylinder { m*<7.685760921185633,2.7193182911058937e-18,-1.7049701696340829>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5}
    cylinder { m*<-4.337997519064569,8.164965809277259,-2.2019807507836546>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5 }
    cylinder {  m*<-4.337997519064569,-8.164965809277259,-2.2019807507836573>, <0.4461265247598325,-5.999562933151155e-18,1.0299749548232944>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    