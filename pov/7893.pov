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
    sphere { m*<-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 1 }        
    sphere {  m*<1.0479832264312892,0.6610318229931944,9.427835278230836>, 1 }
    sphere {  m*<8.415770424754085,0.3759395722009331,-5.1428421508430855>, 1 }
    sphere {  m*<-6.480192768934908,6.899020945821566,-3.6520352476614777>, 1}
    sphere { m*<-4.164292117857636,-8.58955535442786,-2.1779958376019666>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0479832264312892,0.6610318229931944,9.427835278230836>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5 }
    cylinder { m*<8.415770424754085,0.3759395722009331,-5.1428421508430855>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5}
    cylinder { m*<-6.480192768934908,6.899020945821566,-3.6520352476614777>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5 }
    cylinder {  m*<-4.164292117857636,-8.58955535442786,-2.1779958376019666>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5}

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
    sphere { m*<-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 1 }        
    sphere {  m*<1.0479832264312892,0.6610318229931944,9.427835278230836>, 1 }
    sphere {  m*<8.415770424754085,0.3759395722009331,-5.1428421508430855>, 1 }
    sphere {  m*<-6.480192768934908,6.899020945821566,-3.6520352476614777>, 1}
    sphere { m*<-4.164292117857636,-8.58955535442786,-2.1779958376019666>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0479832264312892,0.6610318229931944,9.427835278230836>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5 }
    cylinder { m*<8.415770424754085,0.3759395722009331,-5.1428421508430855>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5}
    cylinder { m*<-6.480192768934908,6.899020945821566,-3.6520352476614777>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5 }
    cylinder {  m*<-4.164292117857636,-8.58955535442786,-2.1779958376019666>, <-0.37118426776887153,-0.3289070908867221,-0.42145481880430324>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    