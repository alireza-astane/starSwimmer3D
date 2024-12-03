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
    sphere { m*<0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 1 }        
    sphere {  m*<0.006341062663738251,-5.7849663403328116e-18,4.1937319331024465>, 1 }
    sphere {  m*<9.406536945226108,6.203178834230739e-20,-2.132492258375737>, 1 }
    sphere {  m*<-4.709201384858897,8.164965809277259,-2.138486764161655>, 1}
    sphere { m*<-4.709201384858897,-8.164965809277259,-2.138486764161658>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.006341062663738251,-5.7849663403328116e-18,4.1937319331024465>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5 }
    cylinder { m*<9.406536945226108,6.203178834230739e-20,-2.132492258375737>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5}
    cylinder { m*<-4.709201384858897,8.164965809277259,-2.138486764161655>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5 }
    cylinder {  m*<-4.709201384858897,-8.164965809277259,-2.138486764161658>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5}

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
    sphere { m*<0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 1 }        
    sphere {  m*<0.006341062663738251,-5.7849663403328116e-18,4.1937319331024465>, 1 }
    sphere {  m*<9.406536945226108,6.203178834230739e-20,-2.132492258375737>, 1 }
    sphere {  m*<-4.709201384858897,8.164965809277259,-2.138486764161655>, 1}
    sphere { m*<-4.709201384858897,-8.164965809277259,-2.138486764161658>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.006341062663738251,-5.7849663403328116e-18,4.1937319331024465>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5 }
    cylinder { m*<9.406536945226108,6.203178834230739e-20,-2.132492258375737>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5}
    cylinder { m*<-4.709201384858897,8.164965809277259,-2.138486764161655>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5 }
    cylinder {  m*<-4.709201384858897,-8.164965809277259,-2.138486764161658>, <0.005631797095349543,-5.780433209785882e-18,1.19373199598464>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    