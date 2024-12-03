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
    sphere { m*<1.2262819013061546,0.09312099630645744,0.5909272135897066>, 1 }        
    sphere {  m*<1.4705179369393697,0.09996096953651151,3.580960687646736>, 1 }
    sphere {  m*<3.9637651260019067,0.09996096953651151,-0.6363215208438813>, 1 }
    sphere {  m*<-3.5298658999232644,7.841936139148581,-2.2212243679212076>, 1}
    sphere { m*<-3.7152737714917623,-8.094572720835444,-2.330161786161116>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4705179369393697,0.09996096953651151,3.580960687646736>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5 }
    cylinder { m*<3.9637651260019067,0.09996096953651151,-0.6363215208438813>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5}
    cylinder { m*<-3.5298658999232644,7.841936139148581,-2.2212243679212076>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5 }
    cylinder {  m*<-3.7152737714917623,-8.094572720835444,-2.330161786161116>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5}

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
    sphere { m*<1.2262819013061546,0.09312099630645744,0.5909272135897066>, 1 }        
    sphere {  m*<1.4705179369393697,0.09996096953651151,3.580960687646736>, 1 }
    sphere {  m*<3.9637651260019067,0.09996096953651151,-0.6363215208438813>, 1 }
    sphere {  m*<-3.5298658999232644,7.841936139148581,-2.2212243679212076>, 1}
    sphere { m*<-3.7152737714917623,-8.094572720835444,-2.330161786161116>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4705179369393697,0.09996096953651151,3.580960687646736>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5 }
    cylinder { m*<3.9637651260019067,0.09996096953651151,-0.6363215208438813>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5}
    cylinder { m*<-3.5298658999232644,7.841936139148581,-2.2212243679212076>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5 }
    cylinder {  m*<-3.7152737714917623,-8.094572720835444,-2.330161786161116>, <1.2262819013061546,0.09312099630645744,0.5909272135897066>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    