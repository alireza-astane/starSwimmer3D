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
    sphere { m*<0.8695367751738077,0.6762146222617651,0.37999563810970016>, 1 }        
    sphere {  m*<1.1129496305448163,0.7341702966928397,3.3695402840457964>, 1 }
    sphere {  m*<3.6061968196073524,0.7341702966928395,-0.8477419244448219>, 1 }
    sphere {  m*<-2.417326534431215,5.709787273789214,-1.563403401985142>, 1}
    sphere { m*<-3.861865856688451,-7.677364128934403,-2.4168443557721284>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1129496305448163,0.7341702966928397,3.3695402840457964>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5 }
    cylinder { m*<3.6061968196073524,0.7341702966928395,-0.8477419244448219>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5}
    cylinder { m*<-2.417326534431215,5.709787273789214,-1.563403401985142>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5 }
    cylinder {  m*<-3.861865856688451,-7.677364128934403,-2.4168443557721284>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5}

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
    sphere { m*<0.8695367751738077,0.6762146222617651,0.37999563810970016>, 1 }        
    sphere {  m*<1.1129496305448163,0.7341702966928397,3.3695402840457964>, 1 }
    sphere {  m*<3.6061968196073524,0.7341702966928395,-0.8477419244448219>, 1 }
    sphere {  m*<-2.417326534431215,5.709787273789214,-1.563403401985142>, 1}
    sphere { m*<-3.861865856688451,-7.677364128934403,-2.4168443557721284>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1129496305448163,0.7341702966928397,3.3695402840457964>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5 }
    cylinder { m*<3.6061968196073524,0.7341702966928395,-0.8477419244448219>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5}
    cylinder { m*<-2.417326534431215,5.709787273789214,-1.563403401985142>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5 }
    cylinder {  m*<-3.861865856688451,-7.677364128934403,-2.4168443557721284>, <0.8695367751738077,0.6762146222617651,0.37999563810970016>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    