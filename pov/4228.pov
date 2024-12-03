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
    sphere { m*<-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 1 }        
    sphere {  m*<0.19748526720848047,0.11036856399630754,4.09511763243544>, 1 }
    sphere {  m*<2.562825765506069,0.014918376094669408,-1.7179967635050402>, 1 }
    sphere {  m*<-1.7934979883930782,2.241358345126894,-1.4627330034698267>, 1}
    sphere { m*<-1.5257107673552464,-2.6463335972770032,-1.273186718307254>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19748526720848047,0.11036856399630754,4.09511763243544>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5 }
    cylinder { m*<2.562825765506069,0.014918376094669408,-1.7179967635050402>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5}
    cylinder { m*<-1.7934979883930782,2.241358345126894,-1.4627330034698267>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5 }
    cylinder {  m*<-1.5257107673552464,-2.6463335972770032,-1.273186718307254>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5}

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
    sphere { m*<-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 1 }        
    sphere {  m*<0.19748526720848047,0.11036856399630754,4.09511763243544>, 1 }
    sphere {  m*<2.562825765506069,0.014918376094669408,-1.7179967635050402>, 1 }
    sphere {  m*<-1.7934979883930782,2.241358345126894,-1.4627330034698267>, 1}
    sphere { m*<-1.5257107673552464,-2.6463335972770032,-1.273186718307254>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19748526720848047,0.11036856399630754,4.09511763243544>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5 }
    cylinder { m*<2.562825765506069,0.014918376094669408,-1.7179967635050402>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5}
    cylinder { m*<-1.7934979883930782,2.241358345126894,-1.4627330034698267>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5 }
    cylinder {  m*<-1.5257107673552464,-2.6463335972770032,-1.273186718307254>, <-0.1718826285001881,-0.08711559929170472,-0.4887872380538569>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    