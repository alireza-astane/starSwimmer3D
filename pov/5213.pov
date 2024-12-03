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
    sphere { m*<-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 1 }        
    sphere {  m*<0.42840018051660306,0.28847310151137556,8.37373326422104>, 1 }
    sphere {  m*<3.5515885469563218,0.0010539123920480509,-3.474806457282676>, 1 }
    sphere {  m*<-2.155649016999239,2.1805335303485136,-2.523297303007771>, 1}
    sphere { m*<-1.8878617959614072,-2.7071584120553838,-2.3337510178452003>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42840018051660306,0.28847310151137556,8.37373326422104>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5 }
    cylinder { m*<3.5515885469563218,0.0010539123920480509,-3.474806457282676>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5}
    cylinder { m*<-2.155649016999239,2.1805335303485136,-2.523297303007771>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5 }
    cylinder {  m*<-1.8878617959614072,-2.7071584120553838,-2.3337510178452003>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5}

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
    sphere { m*<-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 1 }        
    sphere {  m*<0.42840018051660306,0.28847310151137556,8.37373326422104>, 1 }
    sphere {  m*<3.5515885469563218,0.0010539123920480509,-3.474806457282676>, 1 }
    sphere {  m*<-2.155649016999239,2.1805335303485136,-2.523297303007771>, 1}
    sphere { m*<-1.8878617959614072,-2.7071584120553838,-2.3337510178452003>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42840018051660306,0.28847310151137556,8.37373326422104>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5 }
    cylinder { m*<3.5515885469563218,0.0010539123920480509,-3.474806457282676>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5}
    cylinder { m*<-2.155649016999239,2.1805335303485136,-2.523297303007771>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5 }
    cylinder {  m*<-1.8878617959614072,-2.7071584120553838,-2.3337510178452003>, <-0.5211547970625958,-0.1480924611242418,-1.5714964167190646>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    