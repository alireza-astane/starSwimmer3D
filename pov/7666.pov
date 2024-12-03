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
    sphere { m*<-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 1 }        
    sphere {  m*<0.9304493114507816,0.4050659055811141,9.37340678557454>, 1 }
    sphere {  m*<8.29823650977358,0.11997365478885169,-5.1972706434993965>, 1 }
    sphere {  m*<-6.597726683915415,6.6430550284094885,-3.7064637403177887>, 1}
    sphere { m*<-3.6307946682031247,-7.427702078626846,-1.9309398110457605>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9304493114507816,0.4050659055811141,9.37340678557454>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5 }
    cylinder { m*<8.29823650977358,0.11997365478885169,-5.1972706434993965>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5}
    cylinder { m*<-6.597726683915415,6.6430550284094885,-3.7064637403177887>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5 }
    cylinder {  m*<-3.6307946682031247,-7.427702078626846,-1.9309398110457605>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5}

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
    sphere { m*<-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 1 }        
    sphere {  m*<0.9304493114507816,0.4050659055811141,9.37340678557454>, 1 }
    sphere {  m*<8.29823650977358,0.11997365478885169,-5.1972706434993965>, 1 }
    sphere {  m*<-6.597726683915415,6.6430550284094885,-3.7064637403177887>, 1}
    sphere { m*<-3.6307946682031247,-7.427702078626846,-1.9309398110457605>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9304493114507816,0.4050659055811141,9.37340678557454>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5 }
    cylinder { m*<8.29823650977358,0.11997365478885169,-5.1972706434993965>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5}
    cylinder { m*<-6.597726683915415,6.6430550284094885,-3.7064637403177887>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5 }
    cylinder {  m*<-3.6307946682031247,-7.427702078626846,-1.9309398110457605>, <-0.4887181827493807,-0.5848730082988034,-0.47588331146061197>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    