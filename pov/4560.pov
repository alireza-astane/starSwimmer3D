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
    sphere { m*<-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 1 }        
    sphere {  m*<0.3455222780385999,0.18951720024514662,5.932276689410124>, 1 }
    sphere {  m*<2.5243733151962915,-0.0056403948868499615,-2.1951968025584057>, 1 }
    sphere {  m*<-1.8319504387028556,2.220799574145375,-1.9399330425231924>, 1}
    sphere { m*<-1.5641632176650238,-2.666892368258522,-1.7503867573606198>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3455222780385999,0.18951720024514662,5.932276689410124>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5 }
    cylinder { m*<2.5243733151962915,-0.0056403948868499615,-2.1951968025584057>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5}
    cylinder { m*<-1.8319504387028556,2.220799574145375,-1.9399330425231924>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5 }
    cylinder {  m*<-1.5641632176650238,-2.666892368258522,-1.7503867573606198>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5}

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
    sphere { m*<-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 1 }        
    sphere {  m*<0.3455222780385999,0.18951720024514662,5.932276689410124>, 1 }
    sphere {  m*<2.5243733151962915,-0.0056403948868499615,-2.1951968025584057>, 1 }
    sphere {  m*<-1.8319504387028556,2.220799574145375,-1.9399330425231924>, 1}
    sphere { m*<-1.5641632176650238,-2.666892368258522,-1.7503867573606198>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3455222780385999,0.18951720024514662,5.932276689410124>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5 }
    cylinder { m*<2.5243733151962915,-0.0056403948868499615,-2.1951968025584057>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5}
    cylinder { m*<-1.8319504387028556,2.220799574145375,-1.9399330425231924>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5 }
    cylinder {  m*<-1.5641632176650238,-2.666892368258522,-1.7503867573606198>, <-0.2103350788099656,-0.10767437027322418,-0.9659872771072247>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    