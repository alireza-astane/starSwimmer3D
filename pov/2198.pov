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
    sphere { m*<1.1256772646180748,0.2654812153126416,0.531442688813647>, 1 }        
    sphere {  m*<1.3698217230173864,0.28585888538561166,3.521421448999029>, 1 }
    sphere {  m*<3.863068912079923,0.28585888538561166,-0.6958607594915873>, 1 }
    sphere {  m*<-3.222748335852446,7.2340673729152165,-2.039631439045917>, 1}
    sphere { m*<-3.760642033544577,-7.966838641218176,-2.35698880698632>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3698217230173864,0.28585888538561166,3.521421448999029>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5 }
    cylinder { m*<3.863068912079923,0.28585888538561166,-0.6958607594915873>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5}
    cylinder { m*<-3.222748335852446,7.2340673729152165,-2.039631439045917>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5 }
    cylinder {  m*<-3.760642033544577,-7.966838641218176,-2.35698880698632>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5}

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
    sphere { m*<1.1256772646180748,0.2654812153126416,0.531442688813647>, 1 }        
    sphere {  m*<1.3698217230173864,0.28585888538561166,3.521421448999029>, 1 }
    sphere {  m*<3.863068912079923,0.28585888538561166,-0.6958607594915873>, 1 }
    sphere {  m*<-3.222748335852446,7.2340673729152165,-2.039631439045917>, 1}
    sphere { m*<-3.760642033544577,-7.966838641218176,-2.35698880698632>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3698217230173864,0.28585888538561166,3.521421448999029>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5 }
    cylinder { m*<3.863068912079923,0.28585888538561166,-0.6958607594915873>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5}
    cylinder { m*<-3.222748335852446,7.2340673729152165,-2.039631439045917>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5 }
    cylinder {  m*<-3.760642033544577,-7.966838641218176,-2.35698880698632>, <1.1256772646180748,0.2654812153126416,0.531442688813647>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    